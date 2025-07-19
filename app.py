import streamlit as st
from PIL import Image, ImageDraw
import os, json, unicodedata
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 🧠 GCP setup
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
docai_client = documentai.DocumentProcessorServiceClient(
    client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
)

# 🔤 Accent removal + uppercase
def normalize_greek_text(text):
    if not text: return ""
    no_accents = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return no_accents.upper()

# 🧬 Field schema matching
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

# ✂️ Crop to left half
def crop_left(image):
    image = image.convert("RGB")
    w, h = image.size
    return image.crop((0, 0, w // 2, h))

# 🧼 Preprocess with blur
def preprocess(image):
    if image.width > 1500:
        ratio = 1500 / image.width
        image = image.resize((1500, int(image.height * ratio)))
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binarized)

# 📄 Parse with Document AI
def parse_docai(image):
    name = docai_client.processor_path(project_id, location, processor_id)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    raw_doc = documentai.RawDocument(content=buffer.read(), mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw_doc)
    return docai_client.process_document(request=request).document

# 📍 Group + schema + correct
def group_fields(doc, image):
    draw = ImageDraw.Draw(image)
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
            schema_key = map_label_to_schema(label)
            zones[zone].append({
                "Label": label.strip(),
                "Raw Value": value.strip(),
                "Corrected": corrected,
                "Confidence": conf,
                "Schema": schema_key
            })
    return zones

# 🚀 Streamlit UI
st.set_page_config(layout="wide", page_title="Greek Registry Transformer")
st.title("🏛️ Greek Registry OCR — Structured, Accent-Free")

cred = st.sidebar.file_uploader("🔐 Upload GCP credentials", type=["json"])
if cred:
    with open("credentials.json", "wb") as f:
        f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

uploaded = st.file_uploader("📎 Upload registry image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.stop()

original = Image.open(uploaded)
left = crop_left(original)
processed = preprocess(left)

st.image(original, caption="📜 Original Image", use_column_width=True)
st.image(left, caption="◀️ Cropped Left Half", use_column_width=True)
st.image(processed, caption="🧼 Preprocessed", use_column_width=True)

with st.spinner("🔍 Running Document AI..."):
    doc = parse_docai(processed.copy())
    grouped = group_fields(doc, processed.copy())

all_data = []
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
    all_data.extend(output)

st.subheader("📊 Confidence Summary")
st.dataframe(pd.DataFrame(summary, columns=["Form", "Fields", "Avg Confidence"]), use_container_width=True)

st.subheader("💾 Export Structured Data")
df = pd.DataFrame(all_data)
st.download_button("CSV Export", df.to_csv(index=False), "registry_clean.csv", "text/csv")
st.download_button("JSON Export", json.dumps(all_data, indent=2, ensure_ascii=False), "registry_clean.json", "application/json")
