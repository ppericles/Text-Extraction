import streamlit as st
from PIL import Image
import os, json
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 📍 GCP config
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
docai_client = documentai.DocumentProcessorServiceClient(
    client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
)

# 🧼 Preprocess + Resize
def preprocess(image):
    image = image.convert("RGB")
    if image.width > 1000:
        ratio = 1000 / image.width
        image = image.resize((1000, int(image.height * ratio)))
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binarized)

# 🧠 One-shot parse
def parse_docai(pil_img):
    name = docai_client.processor_path(project_id, location, processor_id)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    buffer.seek(0)
    raw_document = documentai.RawDocument(content=buffer.read(), mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = docai_client.process_document(request=request)
    return result.document

# 📊 Assign fields to 3 vertical form zones
def group_fields_by_y(document):
    groups = {1: [], 2: [], 3: []}
    height = document.pages[0].dimension.height
    for page in document.pages:
        for field in page.form_fields:
            value = field.field_value.text_anchor.content or ""
            label = field.field_name.text_anchor.content or ""
            conf = round(field.field_value.confidence * 100, 2)
            box = field.field_value.bounding_poly.normalized_vertices
            avg_y = sum([v.y for v in box]) / len(box)
            if avg_y < 0.33:
                groups[1].append((label.strip(), value.strip(), conf))
            elif avg_y < 0.66:
                groups[2].append((label.strip(), value.strip(), conf))
            else:
                groups[3].append((label.strip(), value.strip(), conf))
    return groups

# ⚙️ UI setup
st.set_page_config(layout="wide", page_title="Greek Registry OCR — Turbo Mode")
st.title("⚡ Greek Registry OCR — 3 Form Parser (Fast)")

cred_file = st.sidebar.file_uploader("🔐 Upload credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

uploaded_file = st.file_uploader("📎 Upload registry scan", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.stop()

image = Image.open(uploaded_file)
preprocessed = preprocess(image)

st.image(image, caption="🖼️ Original Image", use_column_width=True)
st.image(preprocessed, caption="🧼 Preprocessed Image", use_column_width=True)

with st.spinner("🔍 Parsing page with Document AI..."):
    document = parse_docai(preprocessed)
    grouped = group_fields_by_y(document)

all_fields = []

for form_id in [1, 2, 3]:
    st.subheader(f"📄 Φόρμα {form_id}")
    fields = grouped.get(form_id, [])
    if not fields:
        st.info("No fields detected in this section.")
        continue
    edited = []
    for idx, (label, value, conf) in enumerate(fields):
        key = f"form{form_id}_{idx}"
        corrected = st.text_input(f"{label} ({conf}%)", value=value, key=key)
        edited.append({"Form": form_id, "Field": label, "Value": corrected, "Confidence": conf})
    st.dataframe(pd.DataFrame(edited), use_container_width=True)
    all_fields.extend(edited)

# 💾 Export
st.subheader("💾 Export Parsed Data")
df = pd.DataFrame(all_fields)
st.download_button("Download CSV", data=df.to_csv(index=False), file_name="registry_forms.csv", mime="text/csv")
st.download_button("Download JSON", data=json.dumps(all_fields, indent=2, ensure_ascii=False), file_name="registry_forms.json", mime="application/json")
