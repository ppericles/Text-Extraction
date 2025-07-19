import streamlit as st
from PIL import Image
import os, json, unicodedata
import pandas as pd
import numpy as np
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 🧼 Normalize Greek text
def normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFD", text)
    return ''.join(c for c in text if unicodedata.category(c) != "Mn").upper().strip()

# ✂️ Divide image into 3 vertical zones
def split_zones(image):
    w, h = image.size
    thirds = [int(h * i / 3) for i in range(4)]
    zones = []
    for i in range(3):
        crop = image.crop((0, thirds[i], w, thirds[i+1]))
        zones.append(crop.convert("RGB"))
    return zones

# 🔍 Parse each image via Document AI
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
        return client.process_document(request=request).document
    except Exception as e:
        st.error(f"📛 Document AI Error: {e}")
        return None

# 📄 Extract form fields
def extract_fields(doc):
    fields = []
    if not doc or not doc.pages: return fields
    for page in doc.pages:
        for f in page.form_fields:
            label = f.field_name.text_anchor.content or ""
            value = f.field_value.text_anchor.content or ""
            conf = round(f.field_value.confidence * 100, 2)
            fields.append({
                "Label": label.strip(),
                "Raw": value.strip(),
                "Corrected": normalize(value),
                "Confidence": conf,
                "Schema": normalize(label)
            })
    return fields

# 📋 Token-based table reconstruction
def extract_table(doc):
    tokens = []
    for page in doc.pages:
        for token in page.tokens:
            text = token.layout.text_anchor.content or ""
            box = token.layout.bounding_poly.normalized_vertices
            y = sum(v.y for v in box) / len(box)
            x = sum(v.x for v in box) / len(box)
            tokens.append({"text": normalize(text), "y": y, "x": x})

    if not tokens: return []

    # Group tokens into rows by Y position
    tokens.sort(key=lambda t: t["y"])
    rows = []
    current = []
    threshold = 0.01
    for tok in tokens:
        if not current or abs(tok["y"] - current[-1]["y"]) < threshold:
            current.append(tok)
        else:
            rows.append(current)
            current = [tok]
    if current:
        rows.append(current)

    # Require at least 11 rows: 1 header + 10 data
    if len(rows) < 11: return []

    # Align by X to form columns
    header_cells = sorted(rows[0], key=lambda t: t["x"])
    headers = [t["text"] for t in header_cells]
    table = []
    for row in rows[1:11]:
        cells = sorted(row, key=lambda t: t["x"])
        table.append({headers[i]: cells[i]["text"] if i < len(cells) else "" for i in range(len(headers))})
    return table

# 🔧 Streamlit App
st.set_page_config(layout="wide", page_title="Greek Registry Parser")
st.title("🏛️ Registry OCR — 3-Zone Form + Table Extraction")

cred = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred:
    with open("credentials.json", "wb") as f: f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

file = st.file_uploader("📎 Upload Registry Image", type=["jpg", "jpeg", "png"])
if not file: st.stop()

img = Image.open(file)
zones = split_zones(img)
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

forms_all = []
tables_all = []

for i, zone_img in enumerate(zones, start=1):
    st.header(f"📄 Form {i}")
    st.image(zone_img, caption=f"🧾 Cropped Zone {i}", use_column_width=True)

    with st.spinner(f"🔍 Parsing Zone {i}..."):
        doc = parse_docai(zone_img.copy(), project_id, processor_id, location)
        fields = extract_fields(doc)
        table = extract_table(doc)

    # Display form fields
    st.subheader(f"📋 Form {i} Fields")
    if fields:
        forms_all.extend(fields)
        st.dataframe(pd.DataFrame(fields), use_container_width=True)
    else:
        st.warning("⚠️ No form fields detected.")

    # Display table
    st.subheader(f"🧾 Table {i}")
    if table:
        tables_all.extend(table)
        st.dataframe(pd.DataFrame(table), use_container_width=True)
    else:
        st.warning("⚠️ No table rows detected.")

# 💾 Export all data
st.header("💾 Export Parsed Data")

forms_df = pd.DataFrame(forms_all)
st.download_button("📄 Download All Forms CSV", forms_df.to_csv(index=False), "all_forms.csv", "text/csv")
st.download_button("📄 Download All Forms JSON", json.dumps(forms_all, indent=2, ensure_ascii=False), "all_forms.json", "application/json")

if tables_all:
    tables_df = pd.DataFrame(tables_all)
    st.download_button("🧾 Download All Tables CSV", tables_df.to_csv(index=False), "all_tables.csv", "text/csv")
    st.download_button("🧾 Download All Tables JSON", json.dumps(tables_all, indent=2, ensure_ascii=False), "all_tables.json", "application/json")
