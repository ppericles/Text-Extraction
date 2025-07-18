# NOTE: Full code is too long for one message!
# To keep it readable and complete, I’ll paste it in several parts below.

# PART 1: Imports + Setup
import streamlit as st
from PIL import Image, ImageDraw
import os, json
from io import BytesIO
import numpy as np
import pandas as pd
import cv2
from google.cloud import vision
from google.cloud.vision_v1 import types

st.set_page_config(layout="wide", page_title="Greek Registry OCR")
st.title("📜 Greek Registry with Smart Box Detection")

form_ids = [1, 2, 3]
labels_matrix = [
    ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ"],
    ["ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"]
]

# Session state for contours, selections, and results
for key in ["contour_boxes", "selected_boxes", "extracted_values"]:
    if key not in st.session_state:
        st.session_state[key] = {}

# PART 2: File Upload and Reset
cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
uploaded_file = st.file_uploader("📎 Upload registry page", type=["jpg", "jpeg", "png"])
if st.sidebar.button("🔄 Reset All"):
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
            # Detect contours
            gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.bitwise_not(binary)
            scale = 30  # increased to unify partial contours
            horizontal = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1))), cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1)))
            vertical = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (1,scale))), cv2.getStructuringElement(cv2.MORPH_RECT, (1,scale)))
            grid_mask = cv2.bitwise_and(horizontal, vertical)

            contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_w, max_h = crop_np.shape[1], crop_np.shape[0]
            boxes = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w > 40 and h > 20 and w < max_w * 0.95 and h < max_h * 0.95:
                    boxes.append((x, y, w, h))

            # Merge overlapping or close boxes
            def overlap(a, b):
                ax1, ay1, aw, ah = a
                bx1, by1, bw, bh = b
                ax2, ay2 = ax1 + aw, ay1 + ah
                bx2, by2 = bx1 + bw, by1 + bh
                return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

            merged = []
            while boxes:
                base = boxes.pop(0)
                group = [base]
                i = 0
                while i < len(boxes):
                    if overlap(base, boxes[i]):
                        group.append(boxes.pop(i))
                        i = 0
                    else:
                        i += 1
                gx = min(b[0] for b in group)
                gy = min(b[1] for b in group)
                gw = max(b[0] + b[2] for b in group) - gx
                gh = max(b[1] + b[3] for b in group) - gy
                merged.append((gx, gy, gw, gh))
            merged.sort(key=lambda b: (b[1], b[0]))
            st.session_state.contour_boxes[form_key] = merged
                    if form_key not in st.session_state.selected_boxes:
            st.session_state.selected_boxes[form_key] = {}

        selected_boxes = st.session_state.selected_boxes[form_key]
        boxes = st.session_state.contour_boxes[form_key]
        form_data = {}
        field_idx = 0

        for idx, box in enumerate(boxes):
            x, y, w, h = box
            cb_key = f"{form_key}_{idx}"
            if cb_key not in selected_boxes:
                selected_boxes[cb_key] = False
            selected = st.checkbox(f"Box {idx+1} → x={x}, y={y}, w={w}, h={h}",
                                   value=selected_boxes[cb_key],
                                   key=cb_key)
            selected_boxes[cb_key] = selected
            draw.rectangle([(x, y), (x + w, y + h)], outline="purple", width=2)
            draw.text((x + 4, y + 4), f"{idx+1}", fill="purple")

        for idx, box in enumerate(boxes):
            cb_key = f"{form_key}_{idx}"
            if not selected_boxes.get(cb_key):
                continue
            if field_idx >= 8:
                break
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

            # Filter out empty or low-density content
            if len(value.strip()) < 3:
                continue

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
    st.download_button("💾 Download JSON", data=export_json,
                       file_name="registry_data.json", mime="application/json")

    rows = []
    for fid, fields in st.session_state.extracted_values.items():
        row = {"Φόρμα": fid}
        row.update(fields)
        rows.append(row)

    df = pd.DataFrame(rows)
    st.download_button("📤 Download CSV", data=df.to_csv(index=False),
                       file_name="registry_data.csv", mime="text/csv")
