import streamlit as st
from PIL import Image
import os, json, unicodedata
import numpy as np
import pandas as pd
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 🔠 Normalize Greek: Uppercase + Remove Accents
def normalize(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn").upper().strip()

# ✂️ Crop to Left Half
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# 🧼 Optional Preprocess (currently bypassed for diagnosis)
def preprocess(image):
    return image  # You can re-enable blurring if needed later

# 🔍 Document AI Parsing with Diagnostics
def parse_docai(pil_img, project_id, processor_id, location):
    try:
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
        )
        name = client.processor_path(project_id, location, processor_id)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        buf.seek(0)
        raw = documentai.RawDocument(content=buf.read(), mime_type="image/jpeg")
        request = documentai.ProcessRequest(name=name, raw_document=raw)
        result = client.process_document(request=request)
        return result.document
    except Exception as e:
        st.error(f"📛 Document AI Error: {e}")
        return None

# 🧬 Form Field Extraction
def extract_forms(doc):
    zones = {1: [], 2: [], 3: []}
    if not doc or not doc.pages: return zones
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
                "Schema": normalize(label)
            })
    return zones

# 📋 Table Extraction via Header Matching
def extract_table_by_headers(doc, target_headers):
    lines = []
    if not doc or not doc.pages: return []
    for page in doc.pages:
        for para in page.paragraphs:
            content = para.layout.text_anchor.content or ""
            text = normalize(content)
            if text: lines.append(text)

    header_line = next((line for line in lines if all(h in line for h in target_headers)), None)
    if not header_line: return []

    headers = header_line.split()
    rows, collecting = [], False
    for line in lines:
        if line == header_line:
            collecting = True
            continue
        if collecting:
            parts = line.split()
            row = {headers[i]: parts[i] if i < len(parts) else "" for i in range(len(headers))}
            rows.append(row)
            if len(rows) >= 10: break
    return rows

# 🖼️ Streamlit UI
st.set_page_config(layout="wide", page_title="Greek Registry OCR Diagnostic")
st.title("📜 OCR Extractor — Registry Forms + Table")

# 📂 Inputs
cred = st.sidebar.file_uploader("🔐 Upload GCP Credentials (.json)", type=["json"])
if cred:
    with open("credentials.json", "wb") as f: f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

file = st.file_uploader("📎 Upload Registry Scan", type=["jpg", "jpeg", "png"])
if not file: st.stop()

header_input = st.sidebar.text_input("📋 Table Headers (comma-separated)", value="ΗΜΕΡΟΜΗΝΙΑ,ΕΝΕΡΓΕΙΑ,ΣΧΟΛΙΑ")
target_headers = [normalize(h.strip()) for h in header_input.split(",") if h.strip()]

project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

# 🧼 Prep Image
image = Image.open(file)
cropped = crop_left(image)
processed = preprocess(cropped)

st.image(image, caption="📜 Full Original", use_column_width=True)
st.image(processed, caption="🧼 Preprocessed Left Half", use_column_width=True)

doc = parse_docai(processed.copy(), project_id, processor_id, location)
if not doc:
    st.warning("🚫 Parsing failed due to Document AI error.")
    st.stop()

# 📋 Parse Forms
forms = extract_forms(doc)
form_stats, all_fields = [], []
st.subheader("📄 Form Fields")

for zone in [1, 2, 3]:
    fields = forms[zone]
    st.markdown(f"### Ζώνη {zone}")
    if not fields:
        st.info("No fields found.")
        continue
    total = 0
    for i, f in enumerate(fields):
        st.text_input(f"{f['Label']} ({f['Confidence']}%) → [{f['Schema']}]", value=f['Corrected'], key=f"{zone}_{i}")
        total += f["Confidence"]
        all_fields.append(f)
    avg = round(total / len(fields), 2)
    form_stats.append((zone, len(fields), avg))
    st.dataframe(pd.DataFrame(fields), use_container_width=True)

st.subheader("📊 Summary")
st.dataframe(pd.DataFrame(form_stats, columns=["Form", "Fields", "Avg Confidence"]), use_container_width=True)

# 📋 Parse Table
st.subheader("🧾 Table Based on Headers")
table = extract_table_by_headers(doc, target_headers)
if table:
    st.dataframe(pd.DataFrame(table), use_container_width=True)
else:
    st.warning("⚠️ No matching table rows found. Try adjusting headers or review OCR quality.")

# 💾 Export
st.subheader("💾 Export")
form_df = pd.DataFrame(all_fields)
st.download_button("📄 Forms CSV", form_df.to_csv(index=False), "forms.csv", "text/csv")
st.download_button("📄 Forms JSON", json.dumps(all_fields, indent=2, ensure_ascii=False), "forms.json", "application/json")

if table:
    table_df = pd.DataFrame(table)
    st.download_button("🧾 Table CSV", table_df.to_csv(index=False), "table.csv", "text/csv")
    st.download_button("🧾 Table JSON", json.dumps(table, indent=2, ensure_ascii=False), "table.json", "application/json")
