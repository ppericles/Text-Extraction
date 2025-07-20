# ============================================================
# 🔷 BEGIN: Part 1A — Imports, Text Normalization & Validation
# ============================================================

# 📦 Imports
import streamlit as st
import os, json, re, unicodedata
from PIL import Image, ImageDraw
from io import BytesIO
from datetime import datetime
import pandas as pd
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

# 🔡 Normalize Latin/Cyrillic → Greek
def fix_latin_greek(text):
    return "".join({
        "A": "Α", "B": "Β", "E": "Ε", "H": "Η", "K": "Κ", "M": "Μ",
        "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ", "Y": "Υ"
    }.get(c, c) for c in text)

def fix_cyrillic_greek(text):
    return "".join({
        "А": "Α", "В": "Β", "С": "Σ", "Е": "Ε", "Н": "Η",
        "К": "Κ", "Μ": "Μ", "О": "Ο", "Ρ": "Ρ", "Т": "Τ", "Х": "Χ"
    }.get(c, c) for c in text)

def normalize(text):
    if not text: return ""
    text = fix_cyrillic_greek(fix_latin_greek(text))
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\sΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]", "", text)
    return text.upper().strip()

# 📅 Normalize Greek-style dates to DD/MM/YYYY
def normalize_date(text):
    text = text.strip()
    greek_months = {
        "ΙΑΝΟΥΑΡΙΟΥ": "01", "ΦΕΒΡΟΥΑΡΙΟΥ": "02", "ΜΑΡΤΙΟΥ": "03",
        "ΑΠΡΙΛΙΟΥ": "04", "ΜΑΪΟΥ": "05", "ΙΟΥΝΙΟΥ": "06",
        "ΙΟΥΛΙΟΥ": "07", "ΑΥΓΟΥΣΤΟΥ": "08", "ΣΕΠΤΕΜΒΡΙΟΥ": "09",
        "ΟΚΤΩΒΡΙΟΥ": "10", "ΝΟΕΜΒΡΙΟΥ": "11", "ΔΕΚΕΜΒΡΙΟΥ": "12"
    }
    text = normalize(text)
    for gr_month, num in greek_months.items():
        if gr_month in text:
            match = re.search(r"(\d{1,2})\s*" + gr_month + r"\s*(\d{2,4})", text)
            if match:
                d, y = match.group(1), match.group(2)
                y = "19" + y if len(y) == 2 and int(y) >= 30 else "20" + y if len(y) == 2 else y
                return f"{d.zfill(2)}/{num}/{y}"
    if re.match(r"\d{8}$", text):
        return f"{text[:2]}/{text[2:4]}/{text[4:]}"
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(text, fmt)
            y = dt.year
            if y < 1930: y += 2000
            elif y < 2000: y += 1900
            return dt.strftime(f"%d/%m/{y}")
        except: continue
    return text

# 🛡️ Validate metadata field quality
def validate_registry_field(label, text, confidence):
    issues = []
    greek_chars = re.findall(r"[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ]", text or "")
    if not text: issues.append("Missing")
    if label != "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ" and len(greek_chars) < max(3, len(text) // 2):
        issues.append("Non-Greek characters")
    if len(text) < 2: issues.append("Too short")
    if confidence < 50.0: issues.append("Low confidence")
    return issues

# 💡 Suggest fix if value is weak
def suggest_fix(label, text, issues):
    if "Too short" in issues or "Non-Greek characters" in issues:
        fixed = text.title()
        if len(fixed) >= 2 and re.match(r"^[Α-ΩΆΈΉΊΌΎΏ][α-ωάέήίόύώ]{2,}", fixed):
            return fixed
    return None

# 🛡️ Prevent runtime errors by initializing critical variables
def init_context():
    global document, original_image, metadata_rows, detail_rows
    global box_layouts, layout_managers

    if "document" not in globals(): document = None

    if "original_image" not in globals():
        original_image = Image.new("RGB", (800, 1000), color="white")

    if "metadata_rows" not in globals(): metadata_rows = []
    if "detail_rows" not in globals(): detail_rows = []
    if "box_layouts" not in globals(): box_layouts = {}
    if "layout_managers" not in globals(): layout_managers = {}

# 🔧 Initialize context before registry parsing
init_context()
# ============================================================
# 🔷 END: Part 1A
# ============================================================
# ============================================================
# 🔷 BEGIN: Part 1B — Image Processing, Layout & OCR Fallback
# ============================================================

# ✂️ Trim whitespace from image borders
def trim_whitespace(image, threshold=240, buffer=10):
    gray = image.convert("L")
    pixels = gray.load()
    w, h = image.size
    top = next((y for y in range(h) if any(pixels[x, y] < threshold for x in range(w))), 0)
    bottom = next((y for y in reversed(range(h)) if any(pixels[x, y] < threshold for x in range(w))), h)
    left = next((x for x in range(w) if any(pixels[x, y] < threshold for y in range(h))), 0)
    right = next((x for x in reversed(range(w)) if any(pixels[x, y] < threshold for y in range(h))), w)
    return image.crop((max(0, left - buffer), max(0, top - buffer), min(w, right + buffer), min(h, bottom + buffer)))

# ✂️ Crop left half of registry page
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# 🧩 Split image into fixed zones (3 vertical segments)
def split_zones_fixed(image, overlap_px):
    w, h = image.size
    thirds = [int(h * t) for t in [0.0, 0.33, 0.66, 1.0]]
    bounds = [
        (thirds[0], thirds[1] + overlap_px),
        (thirds[1] - overlap_px, thirds[2] + overlap_px),
        (thirds[2] - overlap_px, thirds[3])
    ]
    zones = [image.crop((0, t, w, b)) for t, b in bounds]
    return zones, bounds

# 📐 Convert box between pixel and normalized formats
def convert_box(box, image_size, to_normalized=True):
    if not box or any(v is None for v in box): return (None, None, None, None)
    x, y, w, h = box
    iw, ih = image_size
    return (
        (x / iw, y / ih, w / iw, h / ih) if to_normalized else
        (x * iw, y * ih, w * iw, h * ih)
    )

# 🧠 Estimate OCR confidence based on label type and content
def estimate_confidence(label, text):
    text = text.strip()
    if not text: return 0.0
    if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": return 90.0 if text.isdigit() else 40.0
    if label in ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]:
        return 75.0 if re.match(r"^[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ\s\-]{3,}$", text) else 30.0
    return 50.0

# 🧭 Layout Manager for box geometry (normalized ↔ pixel)
class LayoutManager:
    def __init__(self, image_size):
        self.image_size = image_size

    def to_pixel(self, box):
        return convert_box(box, self.image_size, to_normalized=False)

    def to_normalized(self, box):
        return convert_box(box, self.image_size, to_normalized=True)

    def load_layout(self, layout_dict):
        return {label: self.to_pixel(box) for label, box in layout_dict.items()}

    def save_layout(self, layout_dict):
        return {label: self.to_normalized(box) for label, box in layout_dict.items()}

# 🩹 Vision OCR fallback (used only if layout is present or edited)
def extract_field_from_box_with_vision(pil_img, box, label):
    try:
        x, y, w, h = convert_box(box, pil_img.size, to_normalized=False)
        crop = pil_img.convert("RGB").crop((int(x), int(y), int(x + w), int(y + h)))
        buf = BytesIO(); crop.save(buf, format="JPEG")
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=buf.getvalue())
        response = client.text_detection(image=image, image_context={"language_hints": ["el"]})
        desc = response.text_annotations[0].description.strip() if response.text_annotations else ""
        return desc, estimate_confidence(label, desc)
    except Exception as e:
        st.warning(f"🛑 Vision OCR error for '{label}': {e}")
        return "", 0.0

# 🧵 Extract substring from Document AI TextAnchor
def extract_text_from_anchor(anchor, full_text):
    if not anchor or not anchor.text_segments:
        return ""
    try:
        segments = anchor.text_segments
        return "".join(full_text[int(seg.start_index):int(seg.end_index)] for seg in segments)
    except Exception as e:
        st.warning(f"🧵 TextAnchor extraction failed: {e}")
        return ""

# 📄 Document AI wrapper targeting EU endpoint (v2.16+ SDK)
def parse_docai(pil_image, project_id, processor_id, location):
    try:
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": "eu-documentai.googleapis.com"}
        )
        image_bytes = BytesIO(); pil_image.save(image_bytes, format="JPEG")
        content = image_bytes.getvalue()
        name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
        raw_doc = documentai.RawDocument(content=content, mime_type="image/jpeg")
        request = documentai.ProcessRequest(name=name, raw_document=raw_doc)
        result = client.process_document(request=request)
        return result.document
    except Exception as e:
        st.warning(f"🛑 Document AI error: {e}")
        return None

# ============================================================
# 🔷 END: Part 1B
# ============================================================
# ============================================================
# 🔷 BEGIN: Part 2A — UI, Credentials, Layout & Segmentation
# ============================================================

# 🚀 Streamlit UI setup
st.set_page_config(page_title="📜 Registry Parser", layout="wide")
st.title("📜 Greek Registry Parser — Zone Extraction & Layout Preparation")

# 🎯 Master metadata labels
master_field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ",
    "ΕΠΩΝΥΜΟΝ",
    "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]

# 📦 Session state containers
metadata_rows = []
detail_rows = []
box_layouts = {}
layout_loaded = False

# 🔐 Document AI config
project_id = None
processor_id = None
location = None

# ⚙️ Sidebar controls
st.sidebar.header("⚙️ App Settings")
overlap = st.sidebar.slider("🔁 Zone Overlap (px)", 0, 120, value=40)

# 🔐 Google Credentials upload
cred_file = st.sidebar.file_uploader("🔐 Upload Google Credentials (.json)", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    project_id = "heroic-gantry-380919"
    processor_id = "8f7f56e900fbb37e"
    location = "eu"
    st.sidebar.success("✅ GCP credentials loaded")

# 📥 Optional layout map upload
layout_file = st.sidebar.file_uploader("📥 Import Box Layouts (.json)", type=["json"])
if layout_file:
    try:
        box_layouts = json.load(layout_file)
        layout_loaded = True
        st.sidebar.success(f"📦 Loaded layouts for {len(box_layouts)} zones")
    except Exception as e:
        st.sidebar.error(f"❌ Layout import error: {e}")

# 🖼️ Registry scan upload
uploaded_image = st.file_uploader("📄 Upload Registry Scan", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("📎 Upload a registry page to begin.")
    st.stop()

# ✂️ Image preprocessing and zone segmentation
try:
    original = Image.open(uploaded_image)
    cropped = crop_left(trim_whitespace(original))
    zones, bounds = split_zones_fixed(cropped, overlap)
    if not zones:
        st.error("🚫 Zone segmentation failed.")
        st.stop()
except Exception as e:
    st.error(f"❌ Image preprocessing error: {e}")
    st.stop()

# 📐 LayoutManager setup per zone
layout_managers = {
    str(i + 1): LayoutManager(zones[i].size)
    for i in range(len(zones))
}

# 👀 Display cropped page and preview zones
st.image(cropped, caption="📌 Cropped Registry Page (Left Side)", use_column_width=True)
st.header("🗂️ Zone Previews")
for i, zone in enumerate(zones, start=1):
    st.image(zone, caption=f"Zone {i}", width=300)

# ============================================================
# 🔷 END: Part 2A
# ============================================================
# ============================================================
# 🔷 BEGIN: Part 2B — Registry QA, Table Parsing & Visualization
# ============================================================

st.header("📋 Registry QA & Table Extraction")

# 🧠 Document AI parsing
document = parse_docai(pil_image, project_id, processor_id, location)

# 🩺 Diagnostic check — confirm Document AI success
if not document:
    st.error("🛑 No Document AI response — check your credentials, endpoint, or image input.")
elif not hasattr(document, "tables") or not document.tables:
    st.warning("⚠️ Document AI returned no tables — check image clarity, zone setup, or endpoint configuration.")
else:
    st.success(f"✅ Document AI returned {len(document.tables)} table(s). Proceeding with extraction.")

# 🧾 Parse registry rows from Document AI tables
detail_rows = []
if document and hasattr(document, "tables"):
    for table_index, table in enumerate(document.tables):
        form_id = f"Form_{table_index + 1}"
        for row_index, row in enumerate(table.body_rows):
            row_data = {"FormID": form_id}
            for cell_index, cell in enumerate(row.cells):
                text = extract_text_from_anchor(cell.layout.text_anchor, document.text).strip()
                row_data[f"Column_{cell_index + 1}"] = text
            detail_rows.append(row_data)

    st.success(f"✅ Extracted {len(detail_rows)} registry rows from table(s).")
else:
    st.warning("⚠️ No table rows found to extract.")

# 🖼️ Table Cell Visualization — Boxes + Text
st.subheader("🖼️ Table Cell Preview")

def bounding_poly_to_pixels(poly, image_size):
    if not poly or not poly.vertices: return None
    iw, ih = image_size
    xs = [v.x * iw if v.x is not None else 0 for v in poly.vertices]
    ys = [v.y * ih if v.y is not None else 0 for v in poly.vertices]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return (int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin))  # (x, y, w, h)

def draw_table_boxes_with_text(image, document, full_text):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except:
        font = ImageFont.load_default()

    for table in document.tables:
        for row in table.body_rows:
            for cell in row.cells:
                box = bounding_poly_to_pixels(cell.layout.bounding_poly, image.size)
                if box:
                    x, y, w, h = box
                    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                    cell_text = extract_text_from_anchor(cell.layout.text_anchor, full_text).strip()
                    if cell_text:
                        draw.text((x + 3, y + 3), cell_text[:50], fill="black", font=font)
    return image

if document and hasattr(document, "tables") and document.tables:
    boxed_image = draw_table_boxes_with_text(original_image.copy(), document, document.text)
    st.image(boxed_image, caption="📦 Table Cell Bounding Boxes with Text", use_container_width=True)
else:
    st.info("ℹ️ No table cells available to visualize.")

# ============================================================
# 🔷 END: Part 2B
# ============================================================
# ============================================================
# 🔷 BEGIN: Part 3 — Metadata QA, Export & Layout Packaging
# ============================================================

# 🧼 Filter registry rows to remove empty or sparse entries
def clean_detail_rows(rows, min_filled_fields=2, debug=False):
    cleaned = []
    for row in rows:
        filled = sum(1 for k, v in row.items() if k != "FormID" and v.strip())
        if filled >= min_filled_fields:
            cleaned.append(row)
        elif debug:
            st.warning(f"🧹 Dropped sparse row: {row}")
    return cleaned

# 🧠 Metadata QA Interface
st.header("📊 Metadata Review & Suggestions")
auto_apply = st.checkbox("💡 Auto-apply Suggestions", value=False)

final_metadata = []
for row in metadata_rows:
    fid = row["FormID"]
    st.subheader(f"📄 FormID: {fid}")
    corrected_row = {"FormID": fid}
    for label in master_field_labels:
        if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ":
            continue
        val = row.get(label, "")
        issues = validate_registry_field(label, val, estimate_confidence(label, val))
        suggested = suggest_fix(label, val, issues)
        final_val = suggested if auto_apply and suggested else val
        input_val = st.text_input(f"{label}", value=final_val, key=f"{fid}_{label}")
        corrected_row[label] = input_val
    final_metadata.append(corrected_row)

# 📤 Master Metadata Export
st.header("📤 Export Metadata Table")
df_master = pd.DataFrame(final_metadata)
st.dataframe(df_master, use_container_width=True)

st.download_button("📄 Download Metadata CSV", df_master.to_csv(index=False), "metadata_master.csv", mime="text/csv")
st.download_button("📄 Download Metadata JSON", json.dumps(final_metadata, indent=2, ensure_ascii=False), "metadata_master.json", mime="application/json")

# 📤 Registry Detail Table Export
st.header("📤 Export Registry Table")
cleaned_detail_rows = clean_detail_rows(detail_rows, min_filled_fields=2)
df_detail = pd.DataFrame(cleaned_detail_rows)
st.dataframe(df_detail, use_container_width=True)

st.download_button("📄 Download Registry Table CSV", df_detail.to_csv(index=False), "registry_table.csv", mime="text/csv")
st.download_button("📄 Download Registry Table JSON", json.dumps(cleaned_detail_rows, indent=2, ensure_ascii=False), "registry_table.json", mime="application/json")

# 📑 Column Schema Export
if not df_detail.empty:
    st.subheader("📑 Registry Table Schema")
    schema = list(df_detail.columns)
    st.markdown(f"🧮 Columns: `{', '.join(schema)}`")
    st.download_button("🧾 Download Schema JSON", json.dumps(schema, indent=2, ensure_ascii=False), "registry_table_schema.json", mime="application/json")

# 💾 Layout Export — Normalized & Absolute
st.header("📦 Export Box Layouts")

# Normalized version
st.download_button(
    label="💾 Download Normalized Layouts",
    data=json.dumps(box_layouts, indent=2, ensure_ascii=False),
    file_name="box_layouts_normalized.json",
    mime="application/json"
)

# Absolute pixel coordinates
absolute_layouts = {
    zid: layout_managers[zid].load_layout(boxes)
    for zid, boxes in box_layouts.items()
}
st.download_button(
    label="💾 Download Absolute Layouts",
    data=json.dumps(absolute_layouts, indent=2, ensure_ascii=False),
    file_name="box_layouts_absolute.json",
    mime="application/json"
)

# ============================================================
# 🔷 END: Part 3
# ============================================================
