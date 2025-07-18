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
st.title("📜 Greek Registry OCR — Smart Contour + Hover Review")

form_ids = [1, 2, 3]
labels_matrix = [
    ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ"],
    ["ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"]
]

for key in ["selected_boxes", "extracted_values", "contour_cache"]:
    if key not in st.session_state:
        st.session_state[key] = {}

cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
debug_mode = st.sidebar.checkbox("🧪 Show excluded outer box")
uploaded_file = st.file_uploader("📎 Upload registry page", type=["jpg", "jpeg", "png"])

if st.sidebar.button("🔄 Reset All"):
    st.session_state.selected_boxes = {}
    st.session_state.extracted_values = {}
    st.session_state.contour_cache = {}

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
    pad = 5
    client = vision.ImageAnnotatorClient()

    for form_id in form_ids:
        st.subheader(f"📄 Φόρμα {form_id}")
        y1, y2 = (form_id - 1) * form_height, form_id * form_height
        crop_np = np_image[y1:y2, :].copy()
        st.image(crop_np, caption=f"🖼️ Cropped Φόρμα {form_id}", use_column_width=True)

        preview_img = Image.fromarray(crop_np).convert("RGB")
        draw = ImageDraw.Draw(preview_img)
        form_key = str(form_id)
        cache_key = f"{form_key}_debug_{debug_mode}"

        if cache_key not in st.session_state.contour_cache:
            gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.bitwise_not(binary)
            scale = 30
            horizontal = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1))), cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1)))
            vertical = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (1,scale))), cv2.getStructuringElement(cv2.MORPH_RECT, (1,scale)))
            grid_mask = cv2.bitwise_and(horizontal, vertical)

            contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [(cv2.boundingRect(c)) for c in contours if cv2.boundingRect(c)[2] > 40 and cv2.boundingRect(c)[3] > 20]
            boxes = sorted(boxes, key=lambda b: b[2] * b[3])  # sort by area

            excluded_box = None
            if len(boxes) > 1:
                excluded_box = boxes[-1]
                boxes = boxes[:-1]

            def merge_boxes(boxes, threshold=15):
                merged = []
                while boxes:
                    base = boxes.pop(0)
                    bx, by, bw, bh = base
                    bx2, by2 = bx + bw, by + bh
                    group = [base]
                    i = 0
                    while i < len(boxes):
                        cx, cy, cw, ch = boxes[i]
                        cx2, cy2 = cx + cw, cy + ch
                        if not (bx2 + threshold < cx or cx2 + threshold < bx or
                                by2 + threshold < cy or cy2 + threshold < by):
                            group.append(boxes.pop(i))
                            bx = min(bx, cx)
                            by = min(by, cy)
                            bx2 = max(bx2, cx2)
                            by2 = max(by2, cy2)
                            bw = bx2 - bx
                            bh = by2 - by
                            i = 0
                        else:
                            i += 1
                    merged.append((bx, by, bw, bh))
                return sorted(merged, key=lambda b: (b[1], b[0]))

            st.session_state.contour_cache[cache_key] = merge_boxes(boxes)
            if debug_mode and excluded_box:
                x, y, w, h = excluded_box
                draw.rectangle([(x, y), (x + w, y + h)], outline="gray", width=1)

        if form_key not in st.session_state.selected_boxes:
            st.session_state.selected_boxes[form_key] = {}

        boxes = st.session_state.contour_cache[cache_key]
        selected_boxes = st.session_state.selected_boxes[form_key]
        form_data = {}
        field_idx = 0
        hover_table = []

        for idx, box in enumerate(boxes):
            x, y, w, h = box
            cb_key = f"{form_key}_{idx}"
            if cb_key not in selected_boxes:
                selected_boxes[cb_key] = False
            selected = st.checkbox(f"Box {idx+1} → x={x}, y={y}, w={w}, h={h}", value=selected_boxes[cb_key], key=cb_key)
            selected_boxes[cb_key] = selected
            draw.rectangle([(x, y), (x + w, y + h)], outline="purple", width=2)
            draw.text((x + 4, y + 4), f"{idx+1}", fill="purple")

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
            cleaned = " ".join(text.split("\n")).strip()
            length = len(cleaned)

            hover_table.append({
                "ID": idx+1,
                "Coords": f"x={x}, y={y}, w={w}, h={h}",
                "Chars": length,
                "Content": cleaned if length >= 3 else "(empty)",
                "Selected": "✅" if selected else "—"
            })

            if selected and length >= 3 and field_idx < 8:
                row = field_idx // 4
                col = field_idx % 4
                field = labels_matrix[row][col]
                form_data[field] = cleaned
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                draw.text((x1 + 4, y1 + 4), field, fill="blue")
                field_idx += 1

        st.markdown(f"### ✏️ Review Φόρμα {form_id}")
        for field in labels_matrix[0] + labels_matrix[1]:
            val = form_data.get(field, "")
            corrected = st.text_input(f"{field}", value=val, key=f"{form_key}_{field}")
            st.session_state.extracted_values.setdefault(form_key, {})[field] = corrected

        buffer = BytesIO()
        preview_img.save(buffer, format="PNG")
        buffer.seek(0)
        st.image(np.array(Image.open(buffer)), caption=f"🖼️ Φόρμα {form_id} — Final Grid", use_column_width=True)

        st.markdown("### 🖱️ Hover Box Summary")
        df_hover = pd.DataFrame(hover_table)
        st.dataframe(df_hover, use_container_width=True)

# 💾 Export Final Data
if st.session_state.extracted_values:
    st.markdown("## 💾 Export Final Data")

    export_json = json
