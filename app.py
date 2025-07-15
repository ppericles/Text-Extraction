import streamlit as st
from PIL import Image
import json
import os
from io import BytesIO
import numpy as np
import cv2
from google.cloud import vision
from google.cloud.vision_v1 import types

st.set_page_config(layout="wide", page_title="Greek Registry Cell OCR")
st.title("📜 Greek Registry Cell Extractor")

# === Form IDs
form_ids = [1, 2, 3]

# === Session State Initialization
for key, default in {
    "extracted_values": {i: {} for i in form_ids},
    "click_points": [],
    "ocr_blocks": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# === Sidebar Controls
doc_type = st.sidebar.selectbox("📂 Document Type", ["Registry Page (3 stacked forms)", "Other"])
cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
uploaded_file = st.file_uploader("📎 Upload scanned registry page", type=["jpg", "jpeg", "png"])

if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

# === Image Load
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📄 Scanned Registry Page", use_column_width=True)

# === Registry Form Cell OCR
if uploaded_file and cred_file and doc_type == "Registry Page (3 stacked forms)":
    if st.button("📐 Detect Table Cells in 3 Forms"):
        np_image = np.array(image)
        height, width = np_image.shape[:2]
        form_height = height // 3

        client = vision.ImageAnnotatorClient()

        for form_id in form_ids:
            y_start = (form_id - 1) * form_height
            y_end = y_start + form_height
            form_crop = np_image[y_start:y_end, :]

            st.subheader(f"📄 Φόρμα {form_id}")

            # Preprocess for grid detection
            gray = cv2.cvtColor(form_crop, cv2.COLOR_RGB2GRAY)
            _, bin_img = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
            bin_img = cv2.bitwise_not(bin_img)

            # Line detection kernels
            scale = 20
            horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
            vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))

            horiz_lines = cv2.dilate(cv2.erode(bin_img, horiz_kernel, iterations=1), horiz_kernel, iterations=1)
            vert_lines = cv2.dilate(cv2.erode(bin_img, vert_kernel, iterations=1), vert_kernel, iterations=1)

            grid = cv2.bitwise_and(horiz_lines, vert_lines)
            contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 30 and cv2.boundingRect(c)[3] > 15]
            boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # top-to-bottom, then left-to-right

            form_data = {}

            for i, (x, y, w, h) in enumerate(boxes):
                cell_crop = form_crop[y:y+h, x:x+w]
                pil_crop = Image.fromarray(cell_crop)
                buffer = BytesIO()
                pil_crop.save(buffer, format="JPEG")
                vision_img = types.Image(content=buffer.getvalue())
                response = client.document_text_detection(image=vision_img)
                cell_text = response.full_text_annotation.text.strip()

                form_data[f"Cell {i+1}"] = cell_text
                st.image(pil_crop, caption=f"📎 Φόρμα {form_id} — Κελί {i+1}", width=400)
                st.text_area(f"OCR — Φόρμα {form_id}, Κελί {i+1}", value=cell_text, height=100)

            st.session_state.extracted_values[form_id] = form_data

# === Export Extracted Data
if st.session_state.extracted_values:
    st.markdown("## 💾 Export Extracted Data")
    export_json = json.dumps(st.session_state.extracted_values, indent=2, ensure_ascii=False)
    st.download_button("💾 Export as JSON", data=export_json, file_name="cell_data.json", mime="application/json")
