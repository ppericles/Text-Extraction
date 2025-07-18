import streamlit as st
from PIL import Image
import os, json
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 🔧 Setup
st.set_page_config(layout="wide", page_title="Greek Registry OCR via Document AI")
st.title("📄 Greek Registry Form Parser — with Document AI")

# 🔐 Credential upload
cred_file = st.sidebar.file_uploader("🔐 Upload Google credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

# 📄 Image upload
uploaded_file = st.file_uploader("📎 Upload registry form", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.stop()

# 📍 GCP Details
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

# 🧼 Image preprocessing
def preprocess_image(upload):
    image = Image.open(upload).convert("RGB")
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    _, binarized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_pil = Image.fromarray(binarized).convert("RGB")
    return bin_pil

# 🧠 Parse with Document AI
def parse_with_docai(image):
    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
    )
    name = client.processor_path(project_id, location, processor_id)

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    content = buffer.read()

    raw_document = documentai.RawDocument(content=content, mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    document = result.document
    fields = []

    for page in document.pages:
        for field in page.form_fields:
            key = field.field_name.text_anchor.content.strip() if field.field_name.text_anchor.content else ""
            value = field.field_value.text_anchor.content.strip() if field.field_value.text_anchor.content else ""
            confidence = field.field_value.confidence
            fields.append({
                "Field": key,
                "Value": value,
                "Confidence": round(confidence * 100, 2)
            })
    return fields

# 🖼️ Show original and preprocessed images
original = Image.open(uploaded_file).convert("RGB")
preprocessed = preprocess_image(uploaded_file)
st.image(original, caption="🖼️ Original Scan", use_column_width=True)
st.image(preprocessed, caption="🧼 Preprocessed Image", use_column_width=True)

# 🧠 Extract fields
with st.spinner("🔍 Parsing form with Document AI..."):
    parsed_fields = parse_with_docai(preprocessed)

if not parsed_fields:
    st.warning("⚠️ No fields detected.")
    st.stop()

# ✏️ Editable panel
st.subheader("✏️ Review Extracted Fields")
edited = []
for i, field in enumerate(parsed_fields):
    key = field["Field"] or f"Field {i+1}"
    val = field["Value"]
    confidence = field["Confidence"]
    corrected = st.text_input(f"{key} ({confidence}% confident)", value=val, key=f"field_{i}")
    edited.append({"Field": key, "Value": corrected, "Confidence": confidence})

# 📊 Show table
st.subheader("📊 Extracted Field Summary")
df = pd.DataFrame(edited)
st.dataframe(df, use_container_width=True)

# 💾 Export
st.subheader("💾 Export Results")
json_data = json.dumps(edited, indent=2, ensure_ascii=False)
st.download_button("Download JSON", data=json_data, file_name="parsed_form.json", mime="application/json")
csv_data = df.to_csv(index=False)
st.download_button("Download CSV", data=csv_data, file_name="parsed_form.csv", mime="text/csv")
