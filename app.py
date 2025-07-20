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
        "К": "Κ", "М": "Μ", "О": "Ο", "Р": "Ρ", "Т": "Τ", "Х": "Χ"
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

# 🛡️ Metadata field validator
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
        if len(fixed) >= 2 and re.match(r"^[Α-ΩΆΈΉΊΌΎΏ][α-ωάέήίόύώ]{2,}", fixed): return fixed
    return None
# ✂️ Image trimming and cropping
def trim_whitespace(image, threshold=240, buffer=10):
    gray = image.convert("L")
    pixels = gray.load()
    w, h = image.size
    top = next((y for y in range(h) if any(pixels[x, y] < threshold for x in range(w))), 0)
    bottom = next((y for y in reversed(range(h)) if any(pixels[x, y] < threshold for x in range(w))), h)
    left = next((x for x in range(w) if any(pixels[x, y] < threshold for y in range(h))), 0)
    right = next((x for x in reversed(range(w)) if any(pixels[x, y] < threshold for y in range(h))), w)
    return image.crop((max(0, left - buffer), max(0, top - buffer), min(w, right + buffer), min(h, bottom + buffer)))

def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# 🧩 Fixed zone segmentation (3 horizontal slices with optional overlap)
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

# 📐 Convert box between normalized and pixel coordinates
def convert_box(box, image_size, to_normalized=True):
    if not box or any(v is None for v in box): return (None, None, None, None)
    x, y, w, h = box
    iw, ih = image_size
    return (
        (x / iw, y / ih, w / iw, h / ih) if to_normalized else
        (x * iw, y * ih, w * iw, h * ih)
    )

# 🧠 Heuristic confidence estimation
def estimate_confidence(label, text):
    text = text.strip()
    if not text: return 0.0
    if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": return 90.0 if text.isdigit() else 40.0
    if label in ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]:
        return 75.0 if re.match(r"^[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ\s\-]{3,}$", text) else 30.0
    return 50.0

# 🧭 Layout manager to handle zone-based box geometry
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

# 🩹 Vision OCR fallback (used only if layout_loaded == True)
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
# 🚀 App Config
st.set_page_config(page_title="📜 Registry Parser", layout="wide")
st.title("📜 Greek Registry Parser — Zone Extraction & Preparation")

# 🎯 Master Fields
master_field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ",
    "ΕΠΩΝΥΜΟΝ",
    "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]

# 📦 Containers
metadata_rows = []
detail_rows = []
box_layouts = {}
layout_loaded = False
project_id = None
processor_id = None
location = None

# ⚙️ Sidebar Controls
st.sidebar.header("⚙️ Settings")
overlap = st.sidebar.slider("🔁 Zone Overlap (px)", 0, 120, value=40)

# 🔐 Credentials
cred_file = st.sidebar.file_uploader("🔐 Google Credentials (.json)", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    project_id = "heroic-gantry-380919"
    processor_id = "8f7f56e900fbb37e"
    location = "eu"
    st.sidebar.success("✅ Credentials loaded")

# 📥 Layout Import
layout_file = st.sidebar.file_uploader("📥 Import Layouts (.json)", type=["json"])
if layout_file:
    try:
        box_layouts = json.load(layout_file)
        layout_loaded = True
        st.sidebar.success(f"📦 Loaded layouts for {len(box_layouts)} zones")
    except Exception as e:
        st.sidebar.error(f"❌ Layout import error: {e}")

# 🖼️ Registry Image Upload
uploaded_image = st.file_uploader("📄 Registry Scan", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("📎 Upload a registry scan to begin.")
    st.stop()

# ✂️ Preprocessing & Segmentation
try:
    original = Image.open(uploaded_image)
    cropped = crop_left(trim_whitespace(original))
    zones, bounds = split_zones_fixed(cropped, overlap)
except Exception as e:
    st.error(f"❌ Image preprocessing error: {e}")
    st.stop()

# 🧭 LayoutManager per zone
layout_managers = {
    str(i + 1): LayoutManager(zones[i].size)
    for i in range(len(zones))
}

# 👀 Zone Preview
st.image(cropped, caption="📌 Cropped Page (Left Side)", use_column_width=True)
st.header("🗂️ Zone Previews")
for i, zone in enumerate(zones, start=1):
    st.image(zone, caption=f"Zone {i}", width=300)
# 🩹 Fallback Trigger Helper
def should_use_fallback(layout_dict):
    if layout_loaded:
        return True
    for box in layout_dict.values():
        if box and all(v is not None for v in box):
            return True
    return False

for idx, zone in enumerate(zones, start=1):
    zid = str(idx)
    manager = layout_managers[zid]
    st.header(f"📄 Zone {zid}")

    # 📐 Load editable layout
    layout_pixels = manager.load_layout(box_layouts.get(zid, {}))
    editor_rows = []
    for label in master_field_labels:
        box = layout_pixels.get(label)
        x, y, w, h = box if box else (None, None, None, None)
        editor_rows.append({"Label": label, "X": x, "Y": y, "Width": w, "Height": h})

    editor_df = st.data_editor(
        pd.DataFrame(editor_rows),
        use_container_width=True,
        num_rows="dynamic",
        key=f"editor_{zid}"
    )

    # 💾 Save user edits
    edited_layout = {
        row["Label"]: (row["X"], row["Y"], row["Width"], row["Height"])
        for _, row in editor_df.iterrows()
        if all(v is not None for v in (row["X"], row["Y"], row["Width"], row["Height"]))
    }
    box_layouts[zid] = manager.save_layout(edited_layout)

    # 📌 Overlay for boxes
    overlay = zone.copy()
    draw = ImageDraw.Draw(overlay)
    for label, box in box_layouts[zid].items():
        try:
            x, y, w, h = manager.to_pixel(box)
            draw.rectangle([(int(x), int(y)), (int(x + w), int(y + h))], outline="green", width=2)
            draw.text((int(x), int(y - 14)), label, fill="green")
        except: pass
    st.image(overlay, caption="📌 Master Field Boxes", use_column_width=True)

    # 🧠 Run Document AI
    if not all([project_id, processor_id, location]):
        st.warning("📛 Missing Document AI credentials.")
        continue

    doc = parse_docai(zone.copy(), project_id, processor_id, location)
    full_text = doc.text if doc else ""
    field_map = {
        label: {"Corrected": "", "Confidence": 0.0, "Issues": [], "Suggestion": None}
        for label in master_field_labels
    }

    # 🔍 DocumentAI form fields
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
                    field_map[label_norm] = {
                        "Corrected": corrected,
                        "Confidence": conf,
                        "Issues": issues,
                        "Suggestion": suggestion
                    }

    # 🩹 Vision OCR fallback — layout_loaded OR user-defined boxes
    if should_use_fallback(box_layouts.get(zid, {})):
        for label in master_field_labels:
            if not field_map[label]["Corrected"]:
                box = box_layouts[zid].get(label)
                if box:
                    raw, conf = extract_field_from_box_with_vision(zone, box, label)
                    corrected = normalize(raw)
                    issues = validate_registry_field(label, corrected, conf)
                    suggestion = suggest_fix(label, corrected, issues)
                    field_map[label] = {
                        "Corrected": corrected,
                        "Confidence": conf,
                        "Issues": issues,
                        "Suggestion": suggestion
                    }

    # 🆔 FormID logic
    form_id = field_map["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"]["Corrected"] or f"ZONE_{zid}"
    st.markdown(f"🆔 FormID: `{form_id}`")

    # ✅ Store metadata row
    metadata_rows.append({
        "FormID": form_id,
        **{label: field_map[label]["Corrected"] for label in master_field_labels if label != "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"}
    })

    # 📊 Registry Table Parsing
    expected_columns = [
        "Α/Α",
        "ΤΟΜΟΣ",
        "ΑΡΙΘ.",
        "ΗΜΕΡΟΜΗΝΙΑ ΜΕΤΑΓΡΑΦΗΣ",
        "ΑΡΙΘ. ΕΓΓΡΑΦΟΥ\nΚΑΙ ΕΤΟΣ ΑΥΤΟΥ",
        "ΣΥΜΒΟΛΑΙΟΓΡΑΦΟΣ\nΉ Η ΕΚΔΟΥΣΑ ΑΡΧΗ"
    ]

    if doc:
        for page in doc.pages:
            for table in page.tables:
                st.markdown("📑 Registry Table — Aligned Columns")
                st.markdown(f"🔖 Columns: `{', '.join(expected_columns)}`")

                for row in table.body_rows:
                    row_data = {"FormID": form_id}
                    for i in range(len(expected_columns)):
                        key = expected_columns[i]
                        if i < len(row.cells):
                            cell = row.cells[i]
                            value = extract_text_from_anchor(cell.layout.text_anchor, full_text)
                            if key.strip() == "ΗΜΕΡΟΜΗΝΙΑ ΜΕΤΑΓΡΑΦΗΣ":
                                value = normalize_date(value)
                            row_data[key] = normalize(value)
                        else:
                            row_data[key] = ""
                    detail_rows.append(row_data)
# 🧠 Metadata QA and Manual Fixes
st.header("📊 Metadata Review & Fix Suggestions")
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
        final = suggested if auto_apply and suggested else val
        input_val = st.text_input(f"{label}", value=final, key=f"{fid}_{label}")
        corrected_row[label] = input_val
    final_metadata.append(corrected_row)

# 📤 Metadata Export
st.header("📤 Export Metadata")
df_master = pd.DataFrame(final_metadata)
st.dataframe(df_master, use_container_width=True)

st.download_button("📄 Download Metadata CSV", df_master.to_csv(index=False), "metadata_master.csv", mime="text/csv")
st.download_button("📄 Download Metadata JSON", json.dumps(final_metadata, indent=2, ensure_ascii=False), "metadata_master.json", mime="application/json")

# 📤 Registry Table Export
st.header("📤 Export Registry Table")
df_detail = pd.DataFrame(detail_rows)
st.dataframe(df_detail, use_container_width=True)

st.download_button("📄 Download Registry Table CSV", df_detail.to_csv(index=False), "registry_table.csv", mime="text/csv")
st.download_button("📄 Download Registry Table JSON", json.dumps(detail_rows, indent=2, ensure_ascii=False), "registry_table.json", mime="application/json")

# 📑 Schema Overview
if not df_detail.empty:
    st.subheader("📑 Registry Table Schema")
    schema = list(df_detail.columns)
    st.markdown(f"🧮 Columns: `{', '.join(schema)}`")
    st.download_button("🧾 Download Schema JSON", json.dumps(schema, indent=2, ensure_ascii=False), "registry_table_schema.json", mime="application/json")

# 💾 Layout Export
st.header("📦 Export Layouts")

# Normalized boxes
st.download_button(
    label="💾 Download Normalized Layouts",
    data=json.dumps(box_layouts, indent=2, ensure_ascii=False),
    file_name="box_layouts_normalized.json",
    mime="application/json"
)

# Absolute pixel boxes
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
