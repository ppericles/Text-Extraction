import streamlit as st
from PIL import Image
import os, json, unicodedata
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 🔠 Normalize Greek: Uppercase + Remove Accents
def normalize(text):
    if not text: return ""
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text.upper().strip()

# 🧬 Registry Schema Mapping
schema_map = {
    "ΟΝΟΜΑ": "name",
    "ΟΝΟΜΑΤΕΠΩΝΥΜΟ": "name",
    "ΕΠΩΝΥΜΟ": "surname",
    "ΗΜΕΡΟΜΗΝΙΑ ΓΕΝΝΗΣΗΣ": "birth_date",
    "ΗΜ. ΓΕΝΝΗΣΗΣ": "birth_date",
    "ΤΟΠΟΣ ΓΕΝΝΗΣΗΣ": "birth_place",
    "ΑΡΙΘΜΟΣ ΜΗΤΡΩΟΥ": "registry_id",
    "ΑΡ. ΜΗΤΡΩΟΥ": "registry_id",
    "ΔΙΕΥΘΥΝΣΗ": "address",
    "ΠΟΛΗ": "city"
}

def map_schema(label):
    label_norm = normalize(label)
    return schema_map.get(label_norm, "unknown")

# ✂️ Crop to Left Half
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# 🧼 Preprocess Image
def preprocess(image):
    if image.width > 1500:
        image = image.resize((1500, int(image.height * 1500 / image.width)))
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

# 🔍 Document AI Parsing
def parse_docai(pil_img, project_id, processor_id, location):
    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
    )
    name = client.processor_path(project_id, location, processor_id)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    raw = documentai.RawDocument(content=buf.read(), mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw)
    return client.process_document(request=request).document

# 🧠 Extract Form Fields
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

# 📋 Manual Table Builder Using Header Matching
def extract_table_by_headers(doc, target_headers):
    lines = []
    for page in doc.pages:
        for para in page.paragraphs:
            line = para.layout.text_anchor.content or ""
            line_norm = normalize(line)
            if line_norm: lines.append(line_norm)

    header_found = None
    for line in lines:
        if all(h in line for h in target_headers):
            header_found = line
            break

    if not header_found:
        return []

    headers = header_found.split()
    rows = []
    collecting = False
    for line in lines:
        if line == header_found:
            collecting = True
            continue
        if collecting:
            parts = line.split()
            row = {}
            for i, h in enumerate(headers):
                row[h] = parts[i] if i < len(parts) else ""
            rows.append(row)
            if len(rows) >= 10:
                break
    return rows

# 🖼️ Streamlit App
st.set_page_config(layout="wide", page_title="Registry OCR — Form + Header-Matched Table")
st.title("🏛️ Registry OCR — Forms + Custom Header Table Builder")

# 🔐 Credentials + Upload
cred = st.sidebar.file_uploader("🔐 GCP Credentials (.json)", type=["json"])
if cred:
    with open("credentials.json", "wb") as f:
        f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

file = st.file_uploader("📎 Upload Registry Scan", type=["jpg", "jpeg", "png"])
if not file: st.stop()

headers_input = st.sidebar.text_input("📋 Table Headers (comma-separated in Greek)", value="ΗΜΕΡΟΜΗΝΙΑ,ΕΝΕΡΓΕΙΑ,ΣΧΟΛΙΑ")
target_headers = [normalize(h.strip()) for h in headers_input.split(",") if h.strip()]

project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

# 🧼 Image Prep
img = Image.open(file)
left = crop_left(img)
proc = preprocess(left)

st.image(img, caption="📜 Full Original Image", use_column_width=True)
st.image(proc, caption="🧼 Preprocessed Left Half", use_column_width=True)

with st.spinner("🔍 Parsing image..."):
    doc = parse_docai(proc.copy(), project_id, processor_id, location)
    forms = extract_forms(doc)
    table = extract_table_by_headers(doc, target_headers)

# 📄 Display Forms
st.subheader("📋 Parsed Form Fields")
form_stats, all_fields = [], []

for zone in [1, 2, 3]:
    fields = forms[zone]
    st.markdown(f"### Φόρμα {zone}")
    if not fields:
        st.info("No fields found.")
        continue
    total = 0
    for i, f in enumerate(fields):
        st.text_input(f"{f['Label']} ({f['Confidence']}%) → [{f['Schema']}]", value=f["Corrected"], key=f"{zone}_{i}")
        total += f["Confidence"]
        all_fields.append(f)
    avg = round(total / len(fields), 2)
    form_stats.append((zone, len(fields), avg))
    st.dataframe(pd.DataFrame(fields), use_container_width=True)

st.subheader("📊 Form Summary")
st.dataframe(pd.DataFrame(form_stats, columns=["Form", "Fields", "Avg Confidence"]), use_container_width=True)

# 📋 Table Preview
st.subheader("🧾 Extracted Table Based on Headers")
if table:
    st.dataframe(pd.DataFrame(table), use_container_width=True)
else:
    st.warning("⚠️ Table headers not found — try adjusting header input or checking OCR quality.")

# 💾 Export
st.subheader("💾 Export Data")
form_df = pd.DataFrame(all_fields)
st.download_button("📄 Download Forms CSV", form_df.to_csv(index=False), "forms.csv", "text/csv")
st.download_button("📄 Download Forms JSON", json.dumps(all_fields, indent=2, ensure_ascii=False), "forms.json", "application/json")

if table:
    table_df = pd.DataFrame(table)
    st.download_button("🧾 Download Table CSV", table_df.to_csv(index=False), "table.csv", "text/csv")
    st.download_button("🧾 Download Table JSON", json.dumps(table, indent=2, ensure_ascii=False), "table.json", "application/json")
