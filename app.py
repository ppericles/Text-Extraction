import streamlit as st
from PIL import Image
import os, json
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 🧠 GCP config
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
docai_client = documentai.DocumentProcessorServiceClient(
    client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
)

# 🖼️ Crop image to left half
def crop_left_half(image):
    image = image.convert("RGB")
    width, height = image.size
    left_half = image.crop((0, 0, width // 2, height))
    return left_half

# 🧼 Resize + Preprocess
def preprocess_image(image, apply_denoising=True):
    if image.width > 1500:
        ratio = 1500 / image.width
        image = image.resize((1500, int(image.height * ratio)))
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    if apply_denoising:
        blurred = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    else:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binarized)

# 📄 Parse with Document AI
def parse_docai(pil_img):
    name = docai_client.processor_path(project_id, location, processor_id)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    buffer.seek(0)
    raw_document = documentai.RawDocument(content=buffer.read(), mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = docai_client.process_document(request=request)
    return result.document

# 📍 Group fields into vertical zones
def group_fields(document):
    zones = {1: [], 2: [], 3: []}
    for page in document.pages:
        for field in page.form_fields:
            value = field.field_value.text_anchor.content or ""
            label = field.field_name.text_anchor.content or ""
            conf = round(field.field_value.confidence * 100, 2)
            box = field.field_value.bounding_poly.normalized_vertices
            avg_y = sum(v.y for v in box) / len(box)
            if avg_y < 0.33:
                zones[1].append((label.strip(), value.strip(), conf))
            elif avg_y < 0.66:
                zones[2].append((label.strip(), value.strip(), conf))
            else:
                zones[3].append((label.strip(), value.strip(), conf))
    return zones

# ⚙️ Streamlit UI
st.set_page_config(layout="wide", page_title="Greek Registry OCR — Left-Half Parser")
st.title("🧭 Greek Registry OCR — Left Half Only (Balanced Accuracy)")

cred_file = st.sidebar.file_uploader("🔐 Upload GCP credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

uploaded_file = st.file_uploader("📎 Upload registry image", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.stop()

denoise = st.sidebar.checkbox("🧼 Apply denoising (slower, clearer)", value=True)

# 🖼️ Crop + preprocess
full_img = Image.open(uploaded_file)
left_half = crop_left_half(full_img)
processed = preprocess_image(left_half, apply_denoising=denoise)

st.image(full_img, caption="📜 Full Image (Original)", use_column_width=True)
st.image(left_half, caption="◀️ Cropped to Left Half", use_column_width=True)
st.image(processed, caption="🧼 Preprocessed Left Half", use_column_width=True)

# 🧠 Parse & Group
with st.spinner("🔍 Parsing left half with Document AI..."):
    document = parse_docai(processed)
    grouped = group_fields(document)

all_fields = []
conf_summary = []

for zone in [1, 2, 3]:
    fields = grouped[zone]
    st.subheader(f"📄 Φόρμα {zone}")
    if not fields:
        st.info("No fields detected in this form zone.")
        conf_summary.append((zone, 0, 0))
        continue
    edited = []
    total_conf = 0
    for i, (label, val, conf) in enumerate(fields):
        corrected = st.text_input(f"{label} ({conf}%)", value=val, key=f"{zone}_{i}")
        edited.append({"Form": zone, "Field": label, "Value": corrected, "Confidence": conf})
        total_conf += conf
    avg_conf = round(total_conf / len(fields), 2)
    st.dataframe(pd.DataFrame(edited), use_container_width=True)
    conf_summary.append((zone, len(fields), avg_conf))
    all_fields.extend(edited)

# 📊 Confidence Summary
st.subheader("📊 Extraction Stats")
summary_df = pd.DataFrame(conf_summary, columns=["Form", "Fields Parsed", "Avg Confidence"])
st.dataframe(summary_df, use_container_width=True)

# 💾 Export
st.subheader("💾 Export Results")
df = pd.DataFrame(all_fields)
st.download_button("Download CSV", data=df.to_csv(index=False), file_name="left_half_forms.csv", mime="text/csv")
st.download_button("Download JSON", data=json.dumps(all_fields, indent=2, ensure_ascii=False), file_name="left_half_forms.json", mime="application/json")
