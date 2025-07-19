import streamlit as st
from PIL import Image
import os, json, unicodedata
import pandas as pd
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 🔠 Normalize text: uppercase Greek without accents
def normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFD", text)
    return ''.join(c for c in text if unicodedata.category(c) != "Mn").upper().strip()

# ✂️ Crop image to left half
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# ✂️ Manually tuned vertical slices for 3 zones
def split_zones_tuned(image):
    w, h = image.size
    boundaries = [(0.00, 0.32), (0.33, 0.65), (0.66, 1.00)]
    zones = []
    for top_pct, bottom_pct in boundaries:
        top = int(h * top_pct)
        bottom = int(h * bottom_pct)
        zones.append(image.crop((0, top, w, bottom)).convert("RGB"))
    return zones

# 🔍 Parse zone with Document AI
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

# 📋 Reconstruct table from token geometry
def extract_table(doc):
    tokens = []
    for page in doc.pages:
        for token in page.tokens:
            txt = token.layout.text_anchor.content or ""
            box = token.layout.bounding_poly.normalized_vertices
            y = sum(v.y for v in box) / len(box)
            x = sum(v.x for v in box) / len(box)
            tokens.append({"text": normalize(txt), "y": y, "x": x})

    if not tokens: return []

    tokens.sort(key=lambda t: t["y"])
    rows, current, threshold = [], [], 0.01
    for tok in tokens:
        if not current or abs(tok["y"] - current[-1]["y"]) < threshold:
            current.append(tok)
        else:
            rows.append(current)
            current = [tok]
    if current: rows.append(current)

    if len(rows) < 11: return []

    header_cells = sorted(rows[0], key=lambda t: t["x"])
    headers = [t["text"] for t in header_cells]
    table = []
    for row in rows[1:11]:
        cells = sorted(row, key=lambda t: t["x"])
        table.append({headers[i]: cells[i]["text"] if i < len(cells) else "" for i in range(len(headers))})
    return table

# 🖼️ Streamlit App
st.set_page_config(layout="wide", page_title="Greek Registry Parser")
st.title("🏛️ Registry OCR — Left Half, Tuned 3-Zone Extraction")

cred = st.sidebar.file_uploader("🔐 GCP Credentials (.json)", type=["json"])
if cred:
    with open("credentials.json", "wb") as f: f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

file = st.file_uploader("📎 Upload Registry Image", type=["jpg", "jpeg", "png"])
if not file: st.stop()

img_full = Image.open(file)
img_left = crop_left(img_full)
zones = split_zones_tuned(img_left)

project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

all_fields, all_tables = [], []

for i, zone_img in enumerate(zones, start=1):
    st.header(f"📄 Form {i}")
    st.image(zone_img, caption=f"🧾 Cropped Zone {i} (Left Half)", use_column_width=True)

    with st.spinner(f"🔍 Parsing Form {i}..."):
        doc = parse_docai(zone_img.copy(), project_id, processor_id, location)
        fields = extract_fields(doc)
        table = extract_table(doc)

    st.subheader("📋 Form Fields")
    if fields:
        all_fields.extend(fields)
        st.dataframe(pd.DataFrame(fields), use_container_width=True)
    else:
        st.warning("⚠️ No form fields detected.")

    st.subheader("🧾 Table")
    if table:
        all_tables.extend(table)
        st.dataframe(pd.DataFrame(table), use_container_width=True)
    else:
        st.warning("⚠️ No table rows detected.")

# 💾 Export
st.header("💾 Export Combined Data")

forms_df = pd.DataFrame(all_fields)
st.download_button("📄 Download All Forms CSV", forms_df.to_csv(index=False), "forms.csv", "text/csv")
st.download_button("📄 Download All Forms JSON", json.dumps(all_fields, indent=2, ensure_ascii=False), "forms.json", "application/json")

if all_tables:
    tables_df = pd.DataFrame(all_tables)
    st.download_button("🧾 Download All Tables CSV", tables_df.to_csv(index=False), "tables.csv", "text/csv")
    st.download_button("🧾 Download All Tables JSON", json.dumps(all_tables, indent=2, ensure_ascii=False), "tables.json", "application/json")
