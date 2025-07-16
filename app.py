import streamlit as st
from PIL import Image, ImageDraw
import json, os
from io import BytesIO
import numpy as np
import pandas as pd
import cv2
from google.cloud import vision
from google.cloud.vision_v1 import types

st.set_page_config(layout="wide", page_title="Greek Registry Contour Reviewer")
st.title("📜 Greek Registry OCR with Persistent Selection")

form_ids = [1, 2, 3]
labels_matrix = [
    ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ"],
    ["ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"]
]

# Initialize session state
for key in ["contour_boxes", "selected_boxes", "extracted_values"]:
    if key not in st.session_state:
        st.session_state[key] = {}

cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
uploaded_file = st.file_uploader("📎 Upload registry image", type=["jpg", "jpeg", "png"])
reset_all = st.sidebar.button("🔄 Reset All")

if reset_all:
    st.session_state.contour_boxes = {}
    st.session_state.selected_boxes = {}
    st.session_state.extracted_values = {}

if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📄 Uploaded Registry Page", use_column_width=True)
    np_image = np.array(image)
    height, width = np_image.shape[:2]
    form_height = height // 3
    left_width = width // 2
    pad = 5
    client = vision.ImageAnnotatorClient()

    for form_id in form_ids:
        st.subheader(f"📄 Φόρμα {form_id}")
        y1, y2 = (form_id - 1) * form_height, form_id * form_height
        crop_np = np_image[y1:y2, :left_width].copy()
        preview_img = Image.fromarray(crop_np).convert("RGB")
        draw = ImageDraw.Draw(preview_img)

        form_key = str(form_id)

        if form_key not in st.session_state.contour_boxes:
            # First-time contour detection
            gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.bitwise_not(binary)
            scale = 20
            horizontal = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1))), cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1)))
            vertical = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (1,scale))), cv2.getStructuringElement(cv2.MORPH_RECT, (1,scale)))
            grid_mask = cv2.bitwise_and(horizontal, vertical)

            contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_w, max_h = crop_np.shape[1], crop_np.shape[0]
            boxes = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w > 40 and h > 20 and w < max_w * 0.9 and h < max_h * 0.9:
                    boxes.append((x, y, w, h))
            boxes.sort(key=lambda b: (b[1], b[0]))
            st.session_state.contour_boxes[form_key] = boxes

        if form_key not in st.session_state.selected_boxes:
            st.session_state.selected_boxes[form_key] = {}

        boxes = st.session_state.contour_boxes[form_key]
        form_data = {}
        field_idx = 0

        for idx, box in enumerate(boxes):
            x, y, w, h = box
            cb_key = f"cb_{form_key}_{idx}"
            if cb_key not in st.session_state.selected_boxes[form_key]:
                st.session_state.selected_boxes[form_key][cb_key] = False
            checkbox = st.checkbox(f"Box {idx+1} → x={x}, y={y}, w={w}, h={h}", value=st.session_state.selected_boxes[form_key][cb_key], key=cb_key)
            st.session_state.selected_boxes[form_key][cb_key] = checkbox
            draw.rectangle([(x, y), (x + w, y + h)], outline="purple", width=2)
            draw.text((x + 4, y + 4), f"{idx+1}", fill="purple")

        for idx, box in enumerate(boxes):
            if field_idx >= 8:
                break
            cb_key = f"cb_{form_key}_{idx}"
            if not st.session_state.selected_boxes[form_key].get(cb_key):
                continue
            x, y, w, h = box
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = x + w + pad
            y2 = y + h + pad
            cell = crop_np[y1:y2, x1:x2]
            buffer = BytesIO()
            Image.fromarray(cell).save(buffer, format="JPEG")
            buffer.seek(0)
            vision_img = types.Image(content=buffer.getvalue())
            response = client.document_text_detection(image=vision_img)
            text = response.full_text_annotation.text.strip()
            value = " ".join(text.split("\n")).strip()
            row = field_idx // 4
            col = field_idx % 4
            field = labels_matrix[row][col]
            form_data[field] = value
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
            draw.text((x1 + 4, y1 + 4), field, fill="blue")
            field_idx += 1

        st.session_state.extracted_values[form_key] = form_data

        buffer = BytesIO()
        preview_img.save(buffer, format="PNG")
        buffer.seek(0)
        annotated = Image.open(buffer)
        st.image(np.array(annotated), caption=f"🖼️ Φόρμα {form_id} — Final Grid", use_column_width=True)

        st.markdown(f"### ✏️ Review Φόρμα {form_id}")
        for field in labels_matrix[0] + labels_matrix[1]:
            val = form_data.get(field, "")
            corrected = st.text_input(f"{field}", value=val, key=f"{form_key}_{field}")
            st.session_state.extracted_values[form_key][field] = corrected

if st.session_state.extracted_values:
    st.markdown("## 💾 Export Final Data")
    export_json = json.dumps(st.session_state.extracted_values, indent=2, ensure_ascii=False)
    st.download_button("💾 Download JSON", data=export_json, file_name="registry_data.json", mime="application/json")

    rows = []
    for fid, fields in st.session_state.extracted_values.items():
        row = {"Φόρμα": fid}
        row.update(fields)
        rows.append(row)
    df = pd.DataFrame(rows)
    st.download_button("📤 Download CSV", data=df.to_csv(index=False), file_name="registry_data.csv", mime="text/csv")
