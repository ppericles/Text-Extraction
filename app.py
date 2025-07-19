import streamlit as st
from PIL import Image
import os, json, unicodedata
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 📍 GCP Setup
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
docai_client = documentai.DocumentProcessorServiceClient(
    client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
)

# 🔤 Accent removal + uppercase
def normalize(text):
    if not text: return ""
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text.upper().strip()

# 📐 Schema mapping
schema_map = {
    "ονομα": "name",
    "ονοματεπωνυμο": "name",
    "επωνυμο": "surname",
    "ημερομηνια γεννησης": "birth_date",
    "ημ. γεννησης": "birth_date",
    "τοπος γεννησης": "birth_place",
    "αριθμος μητρωου": "registry_id",
    "αρ. μητρωου": "registry_id",
    "διευθυνση": "address",
    "πολη": "city"
}

def map_schema(label):
    label_norm = normalize(label)
    for raw, key in schema_map.items():
        if raw in label_norm:
            return key
    return "unknown"

# ✂️ Crop to left half
def crop_left(image):
    w, h = image.size
    return image.crop((0, 0, w // 2, h))

# 🧼 Preprocess image
def preprocess(image):
    if image.width > 1500:
        image = image.resize((1500, int(image.height * 1500 / image.width)))
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

# 🔍 Parse via Document AI
def parse_docai(image):
    name = docai_client.processor_path(project_id, location, processor_id)
    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    raw = documentai.RawDocument(content=buf.read(), mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw)
    return docai_client.process_document(request=request).document

# 🧠 Extract form fields into 3 vertical zones
def extract_forms(doc):
    zones = {1: [], 2: [], 3: []}
    for page in doc.pages:
        for field in page.form_fields:
            label = field.field_name.text_anchor.content or ""
            value = field.field_value.text_anchor.content or ""
            conf = round(field.field_value.confidence * 100, 2)
            box = field.field_value.bounding_poly.normalized_vertices
            avg_y = sum(v.y for v in box) / len(box)
            zone = 1 if avg_y < 0.33 else 2 if avg_y < 0.66 else 3
            zones[zone].append({
                "Label": label.strip(),
                "Raw": value.strip(),
                "Corrected": normalize(value),
                "Confidence": conf,
                "Schema": map_schema(label)
            })
    return zones

# 📋 Reconstruct table from OCR layout blocks
def extract_table(doc):
    boxes = []
    for page in doc.pages:
        for token in page.tokens:
            text = token.text
            box = token.layout.bounding_poly.normalized_vertices
            avg_y = sum(v.y for v in box) / len(box)
            avg_x = sum(v.x for v in box) / len(box)
            boxes.append({"text": normalize(text), "y": avg_y, "x": avg_x})
    if not boxes:
        return []

    # Group by Y position (rows)
    boxes.sort(key=lambda b: b["y"])
    rows = []
    current_row = []
    row_threshold = 0.01
    for box in boxes:
        if not current_row or abs(box["y"] - current_row[-1]["y"]) < row_threshold:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    if current_row:
        rows.append(current_row)

    if len(rows) < 2:
        return []

    # First row is header
    header_row = sorted(rows[0], key=lambda b: b["x"])
    headers = [b["text"] for b in header_row]
    table = []
    for row in rows[1:]:
        row_sorted = sorted(row, key=lambda b: b["x"])
        cells = [b["text"] for b in row_sorted]
        row_dict = {}
        for i, h in enumerate(headers):
            row_dict[h] = cells[i] if i < len(cells) else ""
        table.append(row_dict)
    return table

# 🖼️ Streamlit UI
st.set_page_config(layout="wide", page_title="Registry OCR — Form + Table Recovery")
st.title("🏛️ Greek Registry Parser — Schema-Aware + Table Rebuilder")

cred = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred:
    with open("credentials.json", "wb") as f:
        f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("Credentials loaded.")

file = st.file_uploader("📎 Upload Registry Image", type=["jpg", "jpeg", "png"])
if not file: st.stop()

img = Image.open(file)
cropped = crop_left(img)
preprocessed = preprocess(cropped)

st.image(img, caption="📜 Original Full Image", use_column_width=True)
st.image(preprocessed, caption="🧼 Preprocessed Left Half", use_column_width=True)

with st.spinner("🔍 Parsing image via Document AI..."):
    doc = parse_docai(preprocessed.copy())
    forms = extract_forms(doc)
    table_rows = extract_table(doc)

# 📄 Display Forms
st.subheader("📋 Form Field Extraction")
form_summary, form_records = [], []

for zone in [1, 2, 3]:
    fields = forms[zone]
    st.markdown(f"### Φόρμα {zone}")
    if not fields:
        st.info("No fields found.")
        continue
    conf_total = 0
    for i, f in enumerate(fields):
        st.text_input(f"{f['Label']} ({f['Confidence']}%) → [{f['Schema']}]", value=f['Corrected'], key=f"{zone}_{i}")
        conf_total += f['Confidence']
        form_records.append(f)
    avg = round(conf_total / len(fields), 2)
    form_summary.append((zone, len(fields), avg))
    st.dataframe(pd.DataFrame(fields), use_container_width=True)

st.subheader("📊 Form Summary")
st.dataframe(pd.DataFrame(form_summary, columns=["Form", "Fields", "Avg Confidence"]), use_container_width=True)

# 📋 Display Table
st.subheader("🧾 Reconstructed Table (Raw Geometry)")
if table_rows:
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
else:
    st.warning("⚠️ No table rows detected via geometry reconstruction.")

# 💾 Export
st.subheader("💾 Export Parsed Data")
form_df = pd.DataFrame(form_records)
st.download_button("📄 Download Forms CSV", form_df.to_csv(index=False), "forms.csv", "text/csv")
st.download_button("📄 Download Forms JSON", json.dumps(form_records, indent=2, ensure_ascii=False), "forms.json", "application/json")

if table_rows:
    table_df = pd.DataFrame(table_rows)
    st.download_button("🧾 Download Table CSV", table_df.to_csv(index=False), "table.csv", "text/csv")
    st.download_button("🧾 Download Table JSON", json.dumps(table_rows, indent=2, ensure_ascii=False), "table.json", "application/json")
