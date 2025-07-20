# 📦 Imports
import streamlit as st
import os, json, re, unicodedata
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from datetime import datetime
import pandas as pd
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

# 🔡 Normalize Greek text from Latin/Cyrillic
def fix_latin_greek(text):
    return "".join({
        "A": "Α", "B": "Β", "E": "Ε", "H": "Η", "K": "Κ",
        "M": "Μ", "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ",
        "X": "Χ", "Y": "Υ"
    }.get(c, c) for c in text)

def fix_cyrillic_greek(text):
    return "".join({
        "А": "Α", "В": "Β", "С": "Σ", "Е": "Ε", "Н": "Η",
        "К": "Κ", "М": "Μ", "О": "Ο", "Р": "Ρ", "Т": "Τ", "Х": "Χ"
    }.get(c, c) for c in text)

def normalize(text):
    if not text: return ""
    text = fix_cyrillic_greek(fix_latin_greek(text))
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\sΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]", "", text)
    return text.upper().strip()

# 📅 Normalize date formats
def normalize_date(text):
    text = text.strip()
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y"]:
        try:
            return datetime.strptime(text, fmt).strftime("%d/%m/%Y")
        except:
            continue
    return text

# 🛡️ Validate field value
def validate_registry_field(label, corrected_text, confidence):
    issues = []
    greek_chars = re.findall(r"[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ]", corrected_text or "")
    if not corrected_text:
        issues.append("Missing")
    if label != "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ" and len(greek_chars) < max(3, len(corrected_text) // 2):
        issues.append("Non-Greek characters")
    if len(corrected_text) < 2:
        issues.append("Too short")
    if confidence < 50.0:
        issues.append("Low confidence")
    return issues

# 💡 Suggest fix for field
def suggest_fix(label, corrected_text, issues):
    if "Too short" in issues or "Non-Greek characters" in issues:
        fixed = corrected_text.title()
        if len(fixed) >= 2 and re.match(r"^[Α-ΩΆΈΉΊΌΎΏ][α-ωάέήίόύώ]{2,}", fixed):
            return fixed
    return None

# ✂️ Image preprocessing
def trim_whitespace(image, threshold=240, buffer=10):
    gray = image.convert("L")
    pixels = gray.load()
    w, h = image.size
    top = next((y for y in range(h) if any(pixels[x, y] < threshold for x in range(w))), 0)
    bottom = next((y for y in reversed(range(h)) if any(pixels[x, y] < threshold for x in range(w))), h)
    left = next((x for x in range(w) if any(pixels[x, y] < threshold for y in range(h))), 0)
    right = next((x for x in reversed(range(w)) if any(pixels[x, y] < threshold for y in range(h))), w)
    return image.crop((max(0,left-buffer), max(0,top-buffer), min(w,right+buffer), min(h,bottom+buffer)))

def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

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

# 📐 Convert box to/from normalized coordinates
def convert_box(box, image_size, to_normalized=True):
    if not box or any(v is None for v in box):
        return (None, None, None, None)
    x, y, w, h = box
    iw, ih = image_size
    return (
        (x / iw, y / ih, w / iw, h / ih) if to_normalized
        else (x * iw, y * ih, w * iw, h * ih)
    )

# 🧠 Estimate confidence
def estimate_confidence(label, text):
    text = text.strip()
    if not text:
        return 0.0
    if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ":
        return 90.0 if text.isdigit() else 40.0
    if label in ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]:
        return 75.0 if re.match(r"^[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ\s\-]{3,}$", text) else 30.0
    return 50.0

# 🩹 Vision OCR fallback
def extract_field_from_box_with_vision(pil_img, box, label):
    try:
        x, y, w, h = convert_box(box, pil_img.size, to_normalized=False)
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cropped = pil_img.convert("RGB").crop((x1, y1, x2, y2))
        buf = BytesIO(); cropped.save(buf, format="JPEG")
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=buf.getvalue())
        response = client.text_detection(image=image, image_context={"language_hints": ["el"]})
        desc = response.text_annotations[0].description.strip() if response.text_annotations else ""
        return desc, estimate_confidence(label, desc)
    except Exception as e:
        st.warning(f"🛑 Vision OCR error for '{label}': {e}")
        return "", 0.0

# 🧠 Document AI Processor
def parse_docai(pil_img, project_id, processor_id, location):
    try:
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
        )
        name = client.processor_path(project_id, location, processor_id)
        buf = BytesIO(); pil_img.save(buf, format="JPEG")
        raw = documentai.RawDocument(content=buf.getvalue(), mime_type="image/jpeg")
        return client.process_document(request=documentai.ProcessRequest(name=name, raw_document=raw)).document
    except Exception as e:
        st.error(f"📛 Document AI error: {e}")
        return None

# 📄 Resolve text from anchor
def extract_text_from_anchor(anchor, full_text):
    if not anchor or not anchor.text_segments:
        return ""
    return "".join([
        full_text[int(seg.start_index):int(seg.end_index)]
        for seg in anchor.text_segments
        if seg.start_index is not None and seg.end_index is not None
    ]).strip()

# 🧭 LayoutManager class
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
# 🚀 Streamlit Page Config
st.set_page_config(page_title="📜 Greek Registry Parser", layout="wide")
st.title("📜 Registry Parser — Master & Detail Separation")

# 🎯 Field Definitions
master_field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ",
    "ΕΠΩΝΥΜΟΝ",
    "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]

# 📦 App Containers
metadata_rows = []
detail_rows = []
box_layouts = {}

# 🎛️ Sidebar Controls
st.sidebar.header("⚙️ Parser Settings")
overlap = st.sidebar.slider("🔁 Zone Overlap (px)", 0, 120, value=40)

# 🔐 GCP Credential Upload
cred_file = st.sidebar.file_uploader("🔐 GCP Credentials (.json)", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

# 📥 Optional Layout Import
layout_file = st.sidebar.file_uploader("📥 Import Box Layouts (.json)", type=["json"])
if layout_file:
    try:
        box_layouts = json.load(layout_file)
        st.sidebar.success(f"📦 Layouts loaded for {len(box_layouts)} zones")
    except Exception as e:
        st.sidebar.error(f"❌ Layout import failed: {e}")

# 🖼️ Registry Scan Upload
uploaded_image = st.file_uploader("📄 Upload Registry Scan", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("📎 Please upload a registry scan to begin.")
    st.stop()

# ✂️ Image Preprocessing
try:
    original = Image.open(uploaded_image)
    cropped = crop_left(trim_whitespace(original))
    zones, bounds = split_zones_fixed(cropped, overlap)
    if not zones:
        st.error("🚫 No zones detected.")
        st.stop()
except Exception as e:
    st.error(f"❌ Preprocessing error: {e}")
    st.stop()

# 📐 Initialize LayoutManagers
layout_managers = {
    str(i + 1): LayoutManager(zones[i].size)
    for i in range(len(zones))
}

# 👀 Preview Cropped Image and Zones
st.image(cropped, caption="📌 Cropped Registry Page", use_column_width=True)
st.header("🗂️ Zone Previews")
for i, zone in enumerate(zones, start=1):
    st.image(zone, caption=f"Zone {i}", width=300)
# 🔁 Process each zone independently
for idx, zone in enumerate(zones, start=1):
    zid = str(idx)
    manager = layout_managers[zid]

    st.header(f"📄 Processing Zone {zid}")

    # 🛠️ Load or fallback box layout
    if zid not in box_layouts:
        box_layouts[zid] = {
            "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": (0.05, 0.05, 0.15, 0.08),
            "ΕΠΩΝΥΜΟΝ":        (0.05, 0.15, 0.40, 0.07),
            "ΟΝΟΜΑ ΠΑΤΡΟΣ":    (0.05, 0.25, 0.40, 0.07),
            "ΟΝΟΜΑ ΜΗΤΡΟΣ":    (0.05, 0.35, 0.40, 0.07),
            "ΚΥΡΙΟΝ ΟΝΟΜΑ":    (0.05, 0.45, 0.40, 0.07)
        }
        st.warning("⚠️ No layout found — fallback applied.")

    layout_pixels = manager.load_layout(box_layouts[zid])

    # 🧠 OCR with Document AI
    doc = parse_docai(zone.copy(), project_id, processor_id, location)
    full_text = doc.text if doc else ""
    form_fields = {label: {"Raw": "", "Corrected": "", "Confidence": 0.0, "Issues": [], "Suggestion": None}
                   for label in master_field_labels}

    # 🔍 Extract fields via Document AI
    if doc:
        for page in doc.pages:
            for f in page.form_fields:
                label_raw = extract_text_from_anchor(f.field_name.text_anchor, full_text)
                value_raw = extract_text_from_anchor(f.field_value.text_anchor, full_text)
                label_norm = normalize(label_raw)
                if label_norm in master_field_labels:
                    corrected = normalize(value_raw)
                    conf = round(f.field_value.confidence * 100, 2)
                    issues = validate_registry_field(label_norm, corrected, conf)
                    suggestion = suggest_fix(label_norm, corrected, issues)
                    form_fields[label_norm] = {
                        "Raw": value_raw,
                        "Corrected": corrected,
                        "Confidence": conf,
                        "Issues": issues,
                        "Suggestion": suggestion
                    }

    # 🩹 Fill missing fields with Vision OCR
    for label in master_field_labels:
        if not form_fields[label]["Corrected"]:
            box = box_layouts[zid].get(label)
            if box:
                text, conf = extract_field_from_box_with_vision(zone, box, label)
                corrected = normalize(text)
                issues = validate_registry_field(label, corrected, conf)
                suggestion = suggest_fix(label, corrected, issues)
                form_fields[label] = {
                    "Raw": text,
                    "Corrected": corrected,
                    "Confidence": conf,
                    "Issues": issues,
                    "Suggestion": suggestion
                }

    # 🆔 Derive FormID
    form_id = form_fields["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"]["Corrected"]
    if not form_id:
        form_id = f"ZONE_{zid}"
        st.warning(f"⚠️ Missing ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ — using fallback FormID: {form_id}")
    else:
        st.markdown(f"🆔 Detected FormID: `{form_id}`")

    # 📦 Save master row
    metadata_rows.append({
        "FormID": form_id,
        **{label: form_fields[label]["Corrected"] for label in master_field_labels if label != "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"}
    })

    # 🗂️ Parse registry table rows (first 6 columns only)
    if doc:
        for page in doc.pages:
            for table in page.tables:
                headers = []
                for header_row in table.header_rows:
                    for cell in header_row.cells[:6]:
                        raw = extract_text_from_anchor(cell.layout.text_anchor, full_text)
                        headers.append(normalize(raw) or f"COL_{len(headers)}")

                st.markdown(f"📑 Registry Table (Form {form_id}) — Columns: `{', '.join(headers)}`")

                for row in table.body_rows:
                    row_data = {"FormID": form_id}
                    for i in range(min(6, len(row.cells))):
                        cell = row.cells[i]
                        key = headers[i] if i < len(headers) else f"COL_{i}"
                        value = extract_text_from_anchor(cell.layout.text_anchor, full_text)
                        if "ΗΜΕΡ" in key:
                            value = normalize_date(value)
                        row_data[key] = normalize(value)
                    detail_rows.append(row_data)
# 🧠 Final Review & Corrections
st.header("📊 Final Review of Metadata Forms")
apply_suggestions = st.checkbox("💡 Auto-apply Suggested Fixes", value=False)

corrected_metadata_rows = []
for row in metadata_rows:
    form_id = row["FormID"]
    st.subheader(f"📄 FormID: {form_id}")

    corrected_row = {"FormID": form_id}
    for label in master_field_labels:
        if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ":
            continue  # Already used as FormID
        raw = row.get(label, "")
        default = raw
        corrected = st.text_input(f"{label}", value=default, key=f"{form_id}_{label}")
        corrected_row[label] = corrected

    corrected_metadata_rows.append(corrected_row)

# 📤 Export Metadata (Master Table)
st.header("📤 Export Master Metadata Table")
df_master = pd.DataFrame(corrected_metadata_rows)

st.dataframe(df_master, use_container_width=True)

st.download_button(
    label="📄 Download Metadata CSV",
    data=df_master.to_csv(index=False),
    file_name="registry_master_metadata.csv",
    mime="text/csv"
)

st.download_button(
    label="📄 Download Metadata JSON",
    data=json.dumps(corrected_metadata_rows, indent=2, ensure_ascii=False),
    file_name="registry_master_metadata.json",
    mime="application/json"
)

# 📤 Export Detail Table Rows
st.header("📤 Export Registry Detail Table Rows")
df_detail = pd.DataFrame(detail_rows)

st.dataframe(df_detail, use_container_width=True)

st.download_button(
    label="📄 Download Registry Table CSV",
    data=df_detail.to_csv(index=False),
    file_name="registry_detail_rows.csv",
    mime="text/csv"
)

st.download_button(
    label="📄 Download Registry Table JSON",
    data=json.dumps(detail_rows, indent=2, ensure_ascii=False),
    file_name="registry_detail_rows.json",
    mime="application/json"
)

# 📈 Column Schema Preview
if not df_detail.empty:
    st.subheader("📑 Registry Table Schema")
    st.markdown(f"🧮 Columns: `{', '.join(df_detail.columns)}`")
    st.download_button(
        label="🧾 Download Table Schema (JSON)",
        data=json.dumps(list(df_detail.columns), indent=2, ensure_ascii=False),
        file_name="registry_detail_schema.json",
        mime="application/json"
    )

# 💾 Export Box Layouts
st.header("📦 Export Layouts")
st.download_button(
    label="💾 Download Normalized Layouts (JSON)",
    data=json.dumps(box_layouts, indent=2, ensure_ascii=False),
    file_name="box_layouts_normalized.json",
    mime="application/json"
)

absolute_layouts = {
    zid: layout_managers[zid].load_layout(boxes)
    for zid, boxes in box_layouts.items()
}
st.download_button(
    label="💾 Download Absolute Layouts (JSON)",
    data=json.dumps(absolute_layouts, indent=2, ensure_ascii=False),
    file_name="box_layouts_absolute.json",
    mime="application/json"
)
