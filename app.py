import streamlit as st
from PIL import Image, ImageDraw
import json
import os
from io import BytesIO
import numpy as np
import cv2
from google.cloud import vision
from google.cloud.vision_v1 import types

st.set_page_config(layout="wide", page_title="Greek Registry Key-Value Grid")
st.title("📜 Greek Registry Key-Value Grid Parser")

form_ids = [1, 2, 3]
labels_matrix = [
    ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ"],
    ["ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"]
]

if "extracted_values" not in st.session_state:
    st.session_state.extracted_values = {}

cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
uploaded_file = st.file_uploader("📎 Upload registry page", type=["jpg", "jpeg", "png"])

if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📄 Uploaded Registry Page", use_column_width=True)

if uploaded_file and cred_file and st.button("🔍 Parse Forms"):
    np_image = np.array(image)
    h, w = np_image.shape[:2]
    form_h = h // 3
    left_w = w // 2
    client = vision.ImageAnnotatorClient()

    for form_id in form_ids:
        y1 = (form_id - 1) * form_h
        y2 = y1 + form_h
        form_crop = np_image[y1:y2, :left_w].copy()
        gray = cv2.cvtColor(form_crop, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_not(binary)

        # Detect lines
        scale = 20
        horizontal = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1)), iterations=1),
                                cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1)), iterations=1)
        vertical = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale)), iterations=1),
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale)), iterations=1)

        # Find lines
        lines_y = cv2.reduce(horizontal, 1, cv2.REDUCE_MAX).reshape(-1)
        y_coords = [i for i in range(len(lines_y)) if lines_y[i] < 255]
        y_clusters = [y_coords[i] for i in range(0, len(y_coords), max(1, len(y_coords)//3))][:3]

        lines_x = cv2.reduce(vertical, 0, cv2.REDUCE_MAX).reshape(-1)
        x_coords = [i for i in range(len(lines_x)) if lines_x[i] < 255]
        x_clusters = [x_coords[i] for i in range(0, len(x_coords), max(1, len(x_coords)//5))][:5]

        # Build 2x4 grid
        form_data = {}
        preview = Image.fromarray(form_crop)
        draw = ImageDraw.Draw(preview)

        for row in range(2):
            for col in range(4):
                if row >= len(labels_matrix) or col >= len(labels_matrix[row]):
                    continue
                try:
                    y_start = y_clusters[row]
                    y_end = y_clusters[row + 1] if row + 1 < len(y_clusters) else form_crop.shape[0]
                    x_start = x_clusters[col]
                    x_end = x_clusters[col + 1] if col + 1 < len(x_clusters) else form_crop.shape[1]
                    cell_crop = form_crop[y_start:y_end, x_start:x_end]

                    pil_img = Image.fromarray(cell_crop)
                    buffer = BytesIO()
                    pil_img.save(buffer, format="JPEG")
                    vision_img = types.Image(content=buffer.getvalue())
                    response = client.document_text_detection(image=vision_img)
                    raw_text = response.full_text_annotation.text.strip()
                    value = " ".join(raw_text.split("\n")).strip()

                    field = labels_matrix[row][col]
                    form_data[field] = value

                    draw.rectangle([(x_start, y_start), (x_end, y_end)], outline="red", width=2)
                    draw.text((x_start + 5, y_start + 5), f"{field}", fill="blue")
                except Exception as e:
                    form_data[labels_matrix[row][col]] = "🟥 Error"

        st.session_state.extracted_values[str(form_id)] = form_data
        st.subheader(f"📄 Φόρμα {form_id}")
        st.image(preview, caption=f"🔍 Grid Layout — Φόρμα {form_id}", use_column_width=True)

# === Export
if st.session_state.extracted_values:
    st.markdown("## 💾 Export JSON")
    export_json = json.dumps(st.session_state.extracted_values, indent=2, ensure_ascii=False)
    st.download_button("💾 Download JSON", data=export_json, file_name="registry_data.json", mime="application/json")
