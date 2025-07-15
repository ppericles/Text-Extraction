import streamlit as st
from PIL import Image
import json
import os
from io import BytesIO
import numpy as np
import cv2
from google.cloud import vision
from google.cloud.vision_v1 import types

st.set_page_config(layout="wide", page_title="Greek Registry OCR")
st.title("📜 Greek Registry Table OCR")

form_ids = [1, 2, 3]

# === Session state
for key, default in {
    "extracted_values": {i: {} for i in form_ids}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# === Sidebar
cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
uploaded_file = st.file_uploader("📎 Upload registry scan", type=["jpg", "jpeg", "png"])

if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📄 Uploaded Registry Page", use_column_width=True)

# === Modular functions

def detect_cells(image, min_cell_width=40, min_cell_height=20, scale=15):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_not(binary)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))

    horiz_lines = cv2.dilate(cv2.erode(binary, horiz_kernel, iterations=1), horiz_kernel, iterations=1)
    vert_lines = cv2.dilate(cv2.erode(binary, vert_kernel, iterations=1), vert_kernel, iterations=1)

    grid = cv2.bitwise_and(horiz_lines, vert_lines)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > min_cell_width and cv2.boundingRect(c)[3] > min_cell_height]
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # top-down, left-right
    return boxes

def organize_cells_into_grid(cell_boxes, tolerance=10):
    rows = []
    for box in cell_boxes:
        x, y, w, h = box
        inserted = False
        for row in rows:
            if abs(row[0][1] - y) <= tolerance:
                row.append(box)
                inserted = True
                break
        if not inserted:
            rows.append([box])
    for row in rows:
        row.sort(key=lambda b: b[0])
    return rows

def process_form_table(form_crop, client, form_id):
    boxes = detect_cells(form_crop)
    grid = organize_cells_into_grid(boxes)
    result = {}

    for r, row in enumerate(grid):
        for c, (x, y, w, h) in enumerate(row):
            cell_crop = form_crop[y:y+h, x:x+w]
            pil_img = Image.fromarray(cell_crop)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            vision_img = types.Image(content=buffer.getvalue())
            response = client.document_text_detection(image=vision_img)
            text = response.full_text_annotation.text.strip()
            result[(r, c)] = text
    return result

# === Main processing
if uploaded_file and cred_file and st.button("📊 Run Table OCR for All 3 Forms"):
    np_image = np.array(image)
    height = np_image.shape[0]
    form_height = height // 3
    client = vision.ImageAnnotatorClient()

    for form_id in form_ids:
        y1 = (form_id - 1) * form_height
        y2 = y1 + form_height
        form_crop = np_image[y1:y2, :]
        form_data = process_form_table(form_crop, client, form_id)
        st.session_state.extracted_values[form_id] = form_data

        st.subheader(f"📄 Φόρμα {form_id}")
        for (r, c), text in form_data.items():
            st.write(f"🔹 Row {r+1}, Col {c+1}: {text}")

# === Export
if st.session_state.extracted_values:
    st.markdown("## 💾 Export Structured Data")
    export_json = json.dumps(st.session_state.extracted_values, indent=2, ensure_ascii=False)
    st.download_button("💾 Export as JSON", data=export_json, file_name="form_data.json", mime="application/json")
