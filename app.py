import streamlit as st
from PIL import Image, ImageDraw
import json
import os
from io import BytesIO
import numpy as np
import cv2
from google.cloud import vision
from google.cloud.vision_v1 import types
from streamlit_image_coordinates import streamlit_image_coordinates
from difflib import get_close_matches

# === Optional: layout parser
import layout_detector

st.set_page_config(layout="wide", page_title="Greek OCR Annotator")
st.title("🇬🇷 Greek OCR Annotator")

# === Field and Form Setup
field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]
form_ids = [1, 2, 3]

# === Session State Initialization
for key, default in {
    "form_layouts": {i: {} for i in form_ids},
    "ocr_blocks": [],
    "click_points": [],
    "extracted_values": {i: {} for i in form_ids},
    "resolved_forms": set(),
    "current_low_index": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# === Sidebar Controls
view_mode = st.sidebar.radio("🧭 View Mode", ["Tagging", "Compare All Forms"])
form_num = st.sidebar.selectbox("📄 Φόρμα", form_ids)
selected_label = st.sidebar.selectbox("🏷️ Field Label", field_labels)
doc_type = st.sidebar.selectbox("📂 Document Type", ["Registry Book (handwritten)", "Form (printed)", "Report or Paper"])

cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

layout_file = st.sidebar.file_uploader("📂 Import layout (.json)", type=["json"])
if layout_file:
    raw_layout = json.load(layout_file)
    imported = {int(k): v for k, v in raw_layout.items() if k.isdigit()}
    st.session_state.form_layouts.update(imported)
    st.sidebar.success("✅ Layout imported")

uploaded_file = st.file_uploader("📎 Upload scanned form", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

# === Layout Preview
if uploaded_file:
    layout = st.session_state.form_layouts.get(form_num, {})
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for label, box in layout.items():
        draw.rectangle([(box["x1"], box["y1"]), (box["x2"], box["y2"])], outline="green", width=2)
        draw.text((box["x1"], box["y1"] - 10), label, fill="green")
    st.image(preview, caption=f"Φόρμα {form_num} Layout Preview", use_container_width=True)

# === Tagging Interaction
if view_mode == "Tagging" and uploaded_file:
    st.markdown("### 👆 Click twice to tag a field box")
    coords = streamlit_image_coordinates(image)
    if coords:
        st.session_state.click_points.append((coords["x"], coords["y"]))
        st.toast(f"Tagged point {len(st.session_state.click_points)}")

        if len(st.session_state.click_points) == 2:
            (x1, y1), (x2, y2) = st.session_state.click_points
            st.session_state.click_points = []
            box = {
                "x1": min(x1, x2), "y1": min(y1, y2),
                "x2": max(x1, x2), "y2": max(y1, y2)
            }
            label = get_close_matches(selected_label, field_labels, n=1)[0]
            st.session_state.form_layouts.setdefault(form_num, {})[label] = box
            st.success(f"✅ Saved '{label}' for Φόρμα {form_num}")

# === Row-wise OCR with Three-Form Segmentation
if uploaded_file and cred_file and doc_type == "Registry Book (handwritten)":
    if st.button("🧠 Run Row-wise OCR"):
        np_image = np.array(image)
        height, width, _ = np_image.shape
        chunk_width = width // 3

        form_chunks = {
            1: np_image[:, 0:chunk_width],
            2: np_image[:, chunk_width:2*chunk_width],
            3: np_image[:, 2*chunk_width:]
        }

        client = vision.ImageAnnotatorClient()

        for fid, chunk in form_chunks.items():
            st.subheader(f"📄 Φόρμα {fid}")
            gray = cv2.cvtColor(chunk, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY_INV)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rows = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

            extracted_rows = {}

            for i, contour in enumerate(rows):
                x, y, w, h = cv2.boundingRect(contour)
                if h < 30 or w < 150:
                    continue

                crop = chunk[y:y+h, x:x+w]
                pil_crop = Image.fromarray(crop)
                buffer = BytesIO()
                pil_crop.save(buffer, format="JPEG")
                vision_img = types.Image(content=buffer.getvalue())
                response = client.document_text_detection(image=vision_img)
                row_text = response.full_text_annotation.text.strip()

                # === Auto-structure each row into field data
                field_data = {}
                lines = row_text.split("\n")

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if "Επώνυμο" in line or "Επ." in line:
                        field_data["Επώνυμο"] = line.split(":")[-1].strip()
                    elif "Όνομα" in line or "Κύριο Όνομα" in line:
                        field_data["Κύριο Όνομα"] = line.split(":")[-1].strip()
                    elif "Πατρός" in line:
                        field_data["Όνομα Πατρός"] = line.split(":")[-1].strip()
                    elif "Μητρός" in line:
                        field_data["Όνομα Μητρός"] = line.split(":")[-1].strip()
                    elif "Τόπος" in line or "Γεννήσεως" in line:
                        field_data["Τόπος Γεννήσεως"] = line.split(":")[-1].strip()
                    elif any(ch.isdigit() for ch in line) and "19" in line:
                        field_data["Έτος Γεννήσεως"] = line.strip()
                    # Display structured output
                    st.json(field_data)

                extracted_rows[f"Row {i+1}"] = row_text
                st.image(pil_crop, caption=f"📎 Φόρμα {fid} — Row {i+1}", width=600)
                st.text_area(f"OCR — Φόρμα {fid}, Row {i+1}", value=row_text, height=160)

            st.session_state.extracted_values[fid] = extracted_rows

# === Export Tagged Layout
st.markdown("## 💾 Export Tagged Layouts")
st.download_button(
    label="💾 Export as JSON",
    data=json.dumps(st.session_state.form_layouts, indent=2, ensure_ascii=False),
    file_name="form_layouts.json",
    mime="application/json"
)
