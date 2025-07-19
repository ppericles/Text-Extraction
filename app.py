import streamlit as st
from PIL import Image, ImageDraw
import os, json, unicodedata
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# --- CONFIG ---
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
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

# --- UTILS ---
def normalize_text(text):
    if not text: return ""
    no_accents = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return no_accents.upper().strip()

def map_schema(label):
    label_norm = normalize_text(label)
    for raw, schema in schema_map.items():
        if raw in label_norm:
            return schema
    return "unknown"

# --- IMAGE PRE ---
def crop_left(image):
    w, h = image.size
    return image.crop((0, 0, w // 2, h))

def preprocess(image):
    if image.width > 1500:
        image = image.resize((1500, int(image.height * 1500 / image.width)))
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

# --- PARSING ---
def parse_docai(pil_img):
    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
    )
    name = client.processor_path(project_id, location, processor_id)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    raw_doc = documentai.RawDocument(content=buf.getvalue(), mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw_doc)
    return client.process_document(request=request).document

# --- FORM FIELDS ---
def extract_form_fields(doc):
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
                "Corrected": normalize_text(value),
                "Confidence": conf,
                "Schema": map_schema(label)
            })
    return zones

# --- MANUAL TABLE ---
def extract_manual_table(doc):
    blocks = []
    for page in doc.pages:
        for block in page.paragraphs:
            text = block.layout.text_anchor.content or ""
            if not text.strip(): continue
            box = block.layout.bounding_poly.normalized_vertices
            avg_y = sum(v.y for v in box) / len(box)
            blocks.append((avg_y, normalize_text(text)))
    blocks.sort(key=lambda x: x[0])

    # Select last 11 lines as table (1 header + 10 rows)
    table_lines = blocks[-11:] if len(blocks) >= 11 else blocks[-len(blocks):]
    if not table_lines: return []

    header_line = table_lines[0][1]
    headers = [h.strip() for h in header_line.split()]

    rows = []
    for _, line in table_lines[1:]:
        cells = line.split()
        row = {}
        for i, h in enumerate(headers):
            row[h] = cells[i] if i < len(cells) else ""
        rows.append(row)
    return rows

# --- UI ---
st.set_page_config(layout="wide", page_title="Registry Parser with Smart Table")
st.title("🏛️ Greek Registry Parser — Forms + Table (Manual Mode)")

cred = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred:
    with open("credentials.json", "wb") as f:
        f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("Credentials loaded.")

file = st.file_uploader("📎 Upload Registry Scan", type=["jpg", "jpeg", "png"])
if not file: st.stop()

img = Image.open(file)
left_img = crop_left(img)
proc_img = preprocess(left_img)

st.image(img, caption="📜 Full Original Image", use_column_width=True)
st.image(proc_img, caption="🧼 Cropped + Preprocessed", use_column_width=True)

with st.spinner("🔍 Parsing..."):
    doc = parse_docai(proc_img.copy())
    grouped = extract_form_fields(doc)
    table_data = extract_manual_table(doc)

# --- FORM DISPLAY ---
st.subheader("📄 Form Field Extraction")
form_stats, all_fields = [], []

for zone in [1,2,3]:
    fields = grouped[zone]
    st.markdown(f"### 📋 Φόρμα {zone}")
    if not fields:
        st.info("No fields found.")
        continue
    zone_conf = 0
    display = []
    for i, f in enumerate(fields):
        label = f["Label"]
        corrected = f["Corrected"]
        conf = f["Confidence"]
        schema = f["Schema"]
        st.text_input(f"{label} ({conf}%) → [{schema}]", value=corrected, key=f"{zone}_{i}")
        zone_conf += conf
        display.append(f)
    avg = round(zone_conf / len(fields), 2)
    form_stats.append((zone, len(fields), avg))
    all_fields.extend(display)
    st.dataframe(pd.DataFrame(display), use_container_width=True)

st.subheader("📊 Form Stats")
st.dataframe(pd.DataFrame(form_stats, columns=["Form", "Fields", "Avg Confidence"]), use_container_width=True)

# --- TABLE DISPLAY ---
st.subheader("🧾 Manual Table Extraction (Bottom Section)")
if table_data:
    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True)
else:
    st.warning("No table rows were detected from the document blocks.")

# --- EXPORT ---
st.subheader("💾 Export Parsed Data")
form_df = pd.DataFrame(all_fields)
st.download_button("📄 Download Forms CSV", form_df.to_csv(index=False), "parsed_forms.csv", "text/csv")
st.download_button("📄 Download Forms JSON", json.dumps(all_fields, indent=2, ensure_ascii=False), "parsed_forms.json", "application/json")

if table_data:
    table_df = pd.DataFrame(table_data)
    st.download_button("🧾 Download Table CSV", table_df.to_csv(index=False), "parsed_table.csv", "text/csv")
    st.download_button("🧾 Download Table JSON", json.dumps(table_data, indent=2, ensure_ascii=False), "parsed_table.json", "application/json")
