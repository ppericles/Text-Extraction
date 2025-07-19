import streamlit as st
from PIL import Image, ImageDraw
import os, json, unicodedata
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# GCP Setup
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
docai_client = documentai.DocumentProcessorServiceClient(
    client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
)

# Normalize Greek text to uppercase and strip accents
def normalize_greek_text(text):
    if not text: return ""
    no_accents = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return no_accents.upper()

# Schema Mapping
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

def map_label_to_schema(label):
    cleaned = normalize_greek_text(label.strip())
    for raw, key in schema_map.items():
        if raw in cleaned:
            return key
    return "unknown"

# Crop to left half
def crop_left(image):
    image = image.convert("RGB")
    w, h = image.size
    return image.crop((0, 0, w // 2, h))

# Preprocess with Gaussian Blur
def preprocess(image):
    if image.width > 1500:
        image = image.resize((1500, int(image.height * 1500 / image.width)))
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binarized)

# Document AI Parsing
def parse_docai(pil_img):
    name = docai_client.processor_path(project_id, location, processor_id)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    raw_doc = documentai.RawDocument(content=buf.read(), mime_type="image/jpeg")
    req = documentai.ProcessRequest(name=name, raw_document=raw_doc)
    return docai_client.process_document(request=req).document

# Form Field Grouping
def group_form_fields(doc):
    zones = {1: [], 2: [], 3: []}
    for page in doc.pages:
        for field in page.form_fields:
            label = field.field_name.text_anchor.content or ""
            value = field.field_value.text_anchor.content or ""
            conf = round(field.field_value.confidence * 100, 2)
            box = field.field_value.bounding_poly.normalized_vertices
            avg_y = sum(v.y for v in box) / len(box)
            zone = 1 if avg_y < 0.33 else 2 if avg_y < 0.66 else 3
            corrected = normalize_greek_text(value.strip())
            schema = map_label_to_schema(label)
            zones[zone].append({
                "Label": label.strip(),
                "Raw Value": value.strip(),
                "Corrected": corrected,
                "Confidence": conf,
                "Schema": schema
            })
    return zones

# Table Extraction (Assumes bottom 1/3 with header + 10 rows)
def extract_table(doc):
    table_data = []
    for page in doc.pages:
        for table in page.tables:
            headers = []
            for col in table.header_rows[0].cells:
                header = normalize_greek_text(col.layout.text_anchor.content or "").strip()
                headers.append(header)
            for row in table.body_rows:
                row_data = {}
                for i, cell in enumerate(row.cells):
                    value = normalize_greek_text(cell.layout.text_anchor.content or "").strip()
                    key = headers[i] if i < len(headers) else f"col_{i+1}"
                    row_data[key] = value
                table_data.append(row_data)
    return table_data

# Streamlit UI
st.set_page_config(layout="wide", page_title="Greek Registry Parser — Forms & Table")
st.title("🏛️ Greek Registry Parser — Forms + Historic Table")

cred = st.sidebar.file_uploader("🔐 Upload GCP credentials", type=["json"])
if cred:
    with open("credentials.json", "wb") as f:
        f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

uploaded = st.file_uploader("📎 Upload registry image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.stop()

orig = Image.open(uploaded)
left = crop_left(orig)
preproc = preprocess(left)

st.image(orig, caption="📜 Full Image", use_column_width=True)
st.image(preproc, caption="🧼 Preprocessed Left Half", use_column_width=True)

with st.spinner("🔍 Parsing image..."):
    doc = parse_docai(preproc.copy())
    grouped = group_form_fields(doc)
    table_rows = extract_table(doc)

# Display Forms
all_fields = []
summary = []

for zone in [1, 2, 3]:
    fields = grouped.get(zone, [])
    st.subheader(f"📄 Φόρμα {zone}")
    if not fields:
        st.info("No fields found.")
        summary.append((zone, 0, 0))
        continue
    zone_conf = 0
    output = []
    for i, item in enumerate(fields):
        label = item["Label"]
        value = item["Raw Value"]
        corrected = item["Corrected"]
        conf = item["Confidence"]
        schema = item["Schema"]
        st.text_input(f"{label} ({conf}%) → [{schema}]", value=corrected, key=f"{zone}_{i}")
        output.append({
            "Form": zone,
            "Label": label,
            "Raw": value,
            "Corrected": corrected,
            "Confidence": conf,
            "Schema": schema
        })
        zone_conf += conf
    avg = round(zone_conf / len(fields), 2)
    summary.append((zone, len(fields), avg))
    st.dataframe(pd.DataFrame(output), use_container_width=True)
    all_fields.extend(output)

st.subheader("📊 Form Confidence Summary")
st.dataframe(pd.DataFrame(summary, columns=["Form", "Fields", "Avg Confidence"]), use_container_width=True)

# Display Table
st.subheader("🧾 Table Entries (Historic Log)")
if table_rows:
    table_df = pd.DataFrame(table_rows)
    st.dataframe(table_df, use_container_width=True)
else:
    st.info("No table rows found.")

# Export
st.subheader("💾 Export Data")
form_df = pd.DataFrame(all_fields)
st.download_button("Download Forms CSV", form_df.to_csv(index=False), "registry_forms.csv", "text/csv")
st.download_button("Download Forms JSON", json.dumps(all_fields, indent=2, ensure_ascii=False), "registry_forms.json", "application/json")
if table_rows:
    table_df = pd.DataFrame(table_rows)
    st.download_button("Download Table CSV", table_df.to_csv(index=False), "registry_table.csv", "text/csv")
    st.download_button("Download Table JSON", json.dumps(table_rows, indent=2, ensure_ascii=False), "registry_table.json", "application/json")
