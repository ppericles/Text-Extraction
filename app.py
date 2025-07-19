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

# 🔠 Normalize Greek: uppercase + strip accents
def normalize(text):
    if not text: return ""
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text.upper().strip()

# 🧬 Schema Mapping
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

# ✂️ Crop image to left half
def crop_left(image):
    w, h = image.size
    return image.crop((0, 0, w // 2, h))

# 🧼 Preprocess image (rescale + blur + binarize)
def preprocess(image):
    if image.width > 1500:
        image = image.resize((1500, int(image.height * 1500 / image.width)))
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

# 🔍 Parse image via Document AI
def parse_docai(image):
    name = docai_client.processor_path(project_id, location, processor_id)
    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    raw = documentai.RawDocument(content=buf.read(), mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw)
    return docai_client.process_document(request=request).document

# 🧠 Extract form fields
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

# 📋 Table reconstruction via token geometry
def extract_table(doc):
    boxes = []
    for page in doc.pages:
        for token in page.tokens:
            text = token.layout.text_anchor.content or ""
            box = token.layout.bounding_poly.normalized_vertices
            avg_y = sum(v.y for v in box) / len(box)
            avg_x = sum(v.x for v in box) / len(box)
            boxes.append({"text": normalize(text), "y": avg_y, "x": avg_x})
    if not boxes:
        return []

    # Group into rows by Y-position
    boxes.sort(key=lambda b: b["y"])
    rows = []
    current_row = []
    threshold = 0.01
    for box in boxes:
        if not current_row or abs(box["y"] - current_row[-1]["y"]) < threshold:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    if current_row:
        rows.append(current_row)

    if len(rows) < 2:
        return []

    # First row = header
    headers = [b["text"] for b in sorted(rows[0], key=lambda b: b["x"])]
    table = []
    for row in rows[1:]:
        cells = [b["text"] for b in sorted(row, key=lambda b: b["x"])]
        record = {}
        for i, h in enumerate(headers):
            record[h] = cells[i] if i < len(cells) else ""
        table.append(record)
    return table

# 🖼️ Streamlit UI
st.set_page_config(layout="wide", page_title="Registry Parser with Table Recovery")
st.title("🏛️ Greek Registry Parser — Smart Form + Token Table")

cred = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred:
    with open("credentials.json", "wb") as f:
        f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("Credentials loaded")

file = st.file_uploader("📎 Upload Registry Image", type=["jpg", "jpeg", "png"])
if not file: st.stop()

img = Image.open(file)
left = crop_left(img)
proc = preprocess(left)

st.image(img, caption="📜 Full Original Image", use_column_width=True)
st.image(proc, caption="🧼 Preprocessed Left Half", use_column_width=True)

with st.spinner("🔍 Parsing with Document AI..."):
    doc = parse_docai(proc.copy())
    forms = extract_forms(doc)
    table_rows = extract_table(doc)

# 📋 Show Forms
st.subheader("📄 Form Fields (Grouped by Vertical Zones)")
form_stats, all_fields = [], []

for zone in [1, 2, 3]:
    fields = forms[zone]
    st.markdown(f"### Φόρμα {zone}")
    if not fields:
        st.info("No fields found.")
        continue
    total = 0
    for i, f in enumerate(fields):
        st.text_input(f"{f['Label']} ({f['Confidence']}%) → [{f['Schema']}]", value=f['Corrected'], key=f"{zone}_{i}")
        total += f["Confidence"]
        all_fields.append(f)
    avg_conf = round(total / len(fields), 2)
    form_stats.append((zone, len(fields), avg_conf))
    st.dataframe(pd.DataFrame(fields), use_container_width=True)

st.subheader("📊 Form Summary")
st.dataframe(pd.DataFrame(form_stats, columns=["Form", "Fields", "Avg Confidence"]), use_container_width=True)

# 📋 Show Table
st.subheader("🧾 Reconstructed Table (Token Geometry)")
if table_rows:
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
else:
    st.warning("⚠️ No table rows detected.")

# 💾 Export
st.subheader("💾 Export Data")
form_df = pd.DataFrame(all_fields)
st.download_button("Download Forms CSV", form_df.to_csv(index=False), "forms.csv", "text/csv")
st.download_button("Download Forms JSON", json.dumps(all_fields, indent=2, ensure_ascii=False), "forms.json", "application/json")

if table_rows:
    table_df = pd.DataFrame(table_rows)
    st.download_button("Download Table CSV", table_df.to_csv(index=False), "table.csv", "text/csv")
    st.download_button("Download Table JSON", json.dumps(table_rows, indent=2, ensure_ascii=False), "table.json", "application/json")
