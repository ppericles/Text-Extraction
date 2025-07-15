import streamlit as st
from PIL import Image
import json
import os
from io import BytesIO
import numpy as np
import cv2
from google.cloud import vision
from google.cloud.vision_v1 import types

st.set_page_config(layout="wide", page_title="Greek Registry Key-Value Extractor")
st.title("📜 Greek Registry Key-Value Extractor")

form_ids = [1, 2, 3]
labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# === Initialize session state
if "extracted_values" not in st.session_state:
    st.session_state.extracted_values = {}

# === Sidebar Inputs
cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
uploaded_file = st.file_uploader("📎 Upload scanned registry", type=["jpg", "jpeg", "png"])

if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📄 Uploaded Registry Page", use_column_width=True)

# === Main OCR block
if uploaded_file and cred_file and st.button("🔍 Extract Key-Value Pairs"):
    np_image = np.array(image)
    height, width = np_image.shape[:2]
    form_height = height // 3
    left_width = width // 2
    client = vision.ImageAnnotatorClient()

    for form_id in form_ids:
        y1 = (form_id - 1) * form_height
        y2 = y1 + form_height
        form_crop = np_image[y1:y2, :left_width]
        st.subheader(f"📄 Φόρμα {form_id}")

        # Preprocessing for grid detection
        gray = cv2.cvtColor(form_crop, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_not(binary)

        scale = 20
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))
        horiz_lines = cv2.dilate(cv2.erode(binary, horiz_kernel, iterations=1), horiz_kernel, iterations=1)
        vert_lines = cv2.dilate(cv2.erode(binary, vert_kernel, iterations=1), vert_kernel, iterations=1)
        grid = cv2.bitwise_and(horiz_lines, vert_lines)

        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 40 and cv2.boundingRect(c)[3] > 20]
        boxes = sorted(boxes, key=lambda b: (round(b[1] / 10), b[0]))[:8]

        form_data = {}

        for i, (x, y, w, h) in enumerate(boxes):
            if i >= len(labels):
                break
            cell_crop = form_crop[y:y+h, x:x+w]
            pil_img = Image.fromarray(cell_crop)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            vision_img = types.Image(content=buffer.getvalue())
            response = client.document_text_detection(image=vision_img)
            raw_text = response.full_text_annotation.text.strip()
            lines = raw_text.split("\n")

            key = labels[i]
            value = " ".join(lines[1:]).strip() if len(lines) >= 2 else raw_text
            form_data[key] = value

            st.image(cell_crop, caption=f"📎 {key}: {value}", width=400)

        st.session_state.extracted_values[str(form_id)] = form_data

# === Export Results
if st.session_state.extracted_values:
    st.markdown("## 💾 Export Structured Data")
    export_json = json.dumps(st.session_state.extracted_values, indent=2, ensure_ascii=False)
    st.download_button("💾 Export as JSON", data=export_json, file_name="registry_data.json", mime="application/json")
