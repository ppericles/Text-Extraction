import streamlit as st
from PIL import Image
import os, json
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

# 🌐 GCP Config
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

# 🧠 Clients
docai_client = documentai.DocumentProcessorServiceClient(
    client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
)
vision_client = vision.ImageAnnotatorClient()

# 🧼 Preprocessing
def preprocess_image(pil_img):
    np_img = np.array(pil_img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    _, binarized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binarized)

# 📄 Split into 3 form regions
def split_forms(image):
    np_img = np.array(image)
    height = np_img.shape[0]
    thirds = [np_img[i * height // 3:(i + 1) * height // 3, :] for i in range(3)]
    return [Image.fromarray(section).convert("RGB") for section in thirds]

# 🧠 Document AI Parse
def parse_docai(pil_img):
    name = docai_client.processor_path(project_id, location, processor_id)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    buffer.seek(0)
    raw_document = documentai.RawDocument(content=buffer.read(), mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = docai_client.process_document(request=request)
    fields = []
    for page in result.document.pages:
        for field in page.form_fields:
            key = field.field_name.text_anchor.content or ""
            value = field.field_value.text_anchor.content or ""
            confidence = field.field_value.confidence
            fields.append({
                "Field": key.strip(),
                "Value": value.strip(),
                "Confidence": round(confidence * 100, 2)
            })
    return fields

# 📥 Fallback OCR
def parse_with_vision(pil_img):
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    buffer.seek(0)
    image = vision.Image(content=buffer.read())
    response = vision_client.document_text_detection(image=image, image_context={"language_hints": ["el"]})
    text = response.full_text_annotation.text.strip() if response.full_text_annotation.text else ""
    return text

# 📦 UI Setup
st.set_page_config(layout="wide", page_title="Greek Registry OCR — Form Parser")
st.title("📜 Greek Registry OCR — Grouped by Form")

cred_file = st.sidebar.file_uploader("🔐 Upload GCP credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

uploaded_file = st.file_uploader("📎 Upload registry page", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
preprocessed = preprocess_image(image)
form_images = split_forms(preprocessed)

st.image(image, caption="🖼️ Original Image", use_column_width=True)
st.image(preprocessed, caption="🧼 Preprocessed Image", use_column_width=True)

all_forms = []

# 🔎 Parse each form
for idx, form_img in enumerate(form_images):
    form_num = idx + 1
    st.subheader(f"📄 Φόρμα {form_num}")
    st.image(form_img, caption=f"🧾 Cropped Form {form_num}", use_column_width=True)

    with st.spinner(f"🔍 Parsing Form {form_num}..."):
        fields = parse_docai(form_img)

        # Fallback: if no fields, use OCR
        if not fields:
            ocr_text = parse_with_vision(form_img)
            fields = [{"Field": "Full Text OCR", "Value": ocr_text, "Confidence": "Vision"}]

    edited = []
    for i, field in enumerate(fields):
        key = field["Field"] or f"Field {i+1}"
        val = field["Value"]
        confidence = field["Confidence"]
        corrected = st.text_input(f"{key} ({confidence}%)", value=val, key=f"{form_num}_{i}")
        edited.append({"Form": form_num, "Field": key, "Value": corrected, "Confidence": confidence})
    
    st.dataframe(pd.DataFrame(edited), use_container_width=True)
    all_forms.extend(edited)

# 💾 Export
st.subheader("💾 Export All Forms")
df = pd.DataFrame(all_forms)
st.download_button("Download CSV", data=df.to_csv(index=False), file_name="forms.csv", mime="text/csv")
st.download_button("Download JSON", data=json.dumps(all_forms, indent=2, ensure_ascii=False), file_name="forms.json", mime="application/json")
