import streamlit as st
from PIL import Image, ImageDraw
import json
import os
from io import BytesIO
import numpy as np
import pandas as pd
import cv2
from google.cloud import vision
from google.cloud.vision_v1 import types

st.set_page_config(layout="wide", page_title="Greek Registry Grid Debugger")
st.title("📜 Greek Registry Layout + OCR Parser")

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

if uploaded_file and cred_file and st.button("🔍 Parse Forms and Debug Layout"):
    np_image = np.array(image)
    height, width = np_image.shape[:2]
    form_height = height // 3
    left_width = width // 2
    pad = 5
    client = vision.ImageAnnotatorClient()

    for form_id in form_ids:
        y1 = (form_id - 1) * form_height
        y2 = y1 + form_height
        crop_np = np_image[y1:y2, :left_width].copy()
        preview_img = Image.fromarray(crop_np).convert("RGB")
        draw = ImageDraw.Draw(preview_img)

        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_not(binary)

        scale = 20
        horizontal = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1))), cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1)))
        vertical = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (1,scale))), cv2.getStructuringElement(cv2.MORPH_RECT, (1,scale)))
        grid_mask = cv2.bitwise_and(horizontal, vertical)

        st.subheader(f"🧩 Grid Mask Visualization — Φόρμα {form_id}")
        st.image(grid_mask, caption="Grid Mask (AND of horizontal + vertical lines)", use_column_width=True)

        contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 40 and cv2.boundingRect(c)[3] > 20]

        st.markdown(f"**🧮 Contours found: {len(boxes)}**")
        form_data = {}

        if len(boxes) >= 8:
            def cluster_boxes(boxes, row_tol=15):
                rows = []
                for box in sorted(boxes, key=lambda b: b[1]):
                    x, y, w, h = box
                    added = False
                    for r in rows:
                        if abs(r[0][1] - y) <= row_tol:
                            r.append(box)
                            added = True
                            break
                    if not added:
                        rows.append([box])
                for r in rows:
                    r.sort(key=lambda b: b[0])
                return rows

            clustered = cluster_boxes(boxes)

            for row_idx, row in enumerate(clustered[:2]):
                for col_idx, box in enumerate(row[:4]):
                    if row_idx >= 2 or col_idx >= 4:
                        continue
                    x, y, w, h = box
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = x + w + pad, y + h + pad
                    cell = crop_np[y1:y2, x1:x2]
                    buffer = BytesIO()
                    Image.fromarray(cell).save(buffer, format="JPEG")
                    buffer.seek(0)
                    vision_img = types.Image(content=buffer.getvalue())
                    response = client.document_text_detection(image=vision_img)
                    text = response.full_text_annotation.text.strip()
                    value = " ".join(text.split("\n")).strip()
                    field = labels_matrix[row_idx][col_idx]
                    form_data[field] = value
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                    draw.text((x1 + 5, y1 + 5), field, fill="blue")
        else:
            # Draw fallback boxes
            cell_w = crop_np.shape[1] // 4
            cell_h = crop_np.shape[0] // 2
            for r in range(2):
                for c in range(4):
                    x1 = c * cell_w + pad
                    y1 = r * cell_h + pad
                    x2 = (c + 1) * cell_w - pad
                    y2 = (r + 1) * cell_h - pad
                    field = labels_matrix[r][c]
                    form_data[field] = "—"
                    draw.rectangle([(x1, y1), (x2, y2)], outline="orange", width=2)
                    draw.text((x1 + 5, y1 + 5), f"[Fallback] {field}", fill="gray")

        buffer = BytesIO()
        preview_img.save(buffer, format="PNG")
        buffer.seek(0)
        annotated = Image.open(buffer)
        st.image(np.array(annotated), caption=f"🖼️ Φόρμα {form_id} — Annotated Grid", use_column_width=True)

        st.session_state.extracted_values[str(form_id)] = form_data

        st.markdown(f"### ✏️ Review Φόρμα {form_id}")
        for field in labels_matrix[0] + labels_matrix[1]:
            val = form_data.get(field, "")
            corrected = st.text_input(f"{field}", value=val, key=f"{form_id}_{field}")
            st.session_state.extracted_values[str(form_id)][field] = corrected

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
