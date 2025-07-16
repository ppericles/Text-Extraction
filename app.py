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
st.title("📜 Greek Registry Key-Value Grid Extractor")

form_ids = [1, 2, 3]
labels_matrix = [
    ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ"],
    ["ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"]
]

# === Session state
if "extracted_values" not in st.session_state:
    st.session_state.extracted_values = {}

# === Sidebar
cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
uploaded_file = st.file_uploader("📎 Upload registry page", type=["jpg", "jpeg", "png"])

if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📄 Uploaded Registry", use_column_width=True)

# === Grid-based OCR
if uploaded_file and cred_file and st.button("🔍 Extract Key-Value Grid"):
    np_image = np.array(image)
    height, width = np_image.shape[:2]
    form_height = height // 3
    left_width = width // 2
    client = vision.ImageAnnotatorClient()

    for form_id in form_ids:
        y1 = (form_id - 1) * form_height
        y2 = y1 + form_height
        form_crop = np_image[y1:y2, :left_width].copy()
        st.subheader(f"📄 Φόρμα {form_id}")

        # Preprocessing
        gray = cv2.cvtColor(form_crop, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_not(binary)

        scale = 20
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))
        horiz_lines = cv2.dilate(cv2.erode(binary, horiz_kernel, iterations=1), horiz_kernel, iterations=1)
        vert_lines = cv2.dilate(cv2.erode(binary, vert_kernel, iterations=1), vert_kernel, iterations=1)
        grid_mask = cv2.bitwise_and(horiz_lines, vert_lines)

        contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 40 and cv2.boundingRect(c)[3] > 20]

        # Organize into rows (2) and columns (4)
        def cluster_boxes(boxes, row_tol=15, col_tol=15):
            rows = []
            for box in sorted(boxes, key=lambda b: b[1]):
                x, y, w, h = box
                inserted = False
                for r in rows:
                    if abs(r[0][1] - y) <= row_tol:
                        r.append(box)
                        inserted = True
                        break
                if not inserted:
                    rows.append([box])
            for r in rows:
                r.sort(key=lambda b: b[0])  # left to right
            return rows

        clustered = cluster_boxes(boxes)
        form_data = {}

        for row_idx, row in enumerate(clustered[:2]):
            for col_idx, (x, y, w, h) in enumerate(row[:4]):
                cell_crop = form_crop[y:y+h, x:x+w]
                pil_img = Image.fromarray(cell_crop)
                buffer = BytesIO()
                pil_img.save(buffer, format="JPEG")
                vision_img = types.Image(content=buffer.getvalue())
                response = client.document_text_detection(image=vision_img)
                raw_text = response.full_text_annotation.text.strip()
                lines = raw_text.split("\n")
                value = " ".join(lines).strip()
                field = labels_matrix[row_idx][col_idx]
                form_data[field] = value

                st.image(cell_crop, caption=f"📎 {field}: {value}", width=400)

        st.session_state.extracted_values[str(form_id)] = form_data

# === Export
if st.session_state.extracted_values:
    st.markdown("## 💾 Export Structured Data")
    export_json = json.dumps(st.session_state.extracted_values, indent=2, ensure_ascii=False)
    st.download_button("💾 Export JSON", data=export_json, file_name="registry_data.json", mime="application/json")
