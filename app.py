import streamlit as st
from PIL import Image, ImageDraw
import json
import os
from io import BytesIO
import numpy as np
import pandas as pd
import cv2
from google.cloud import vision

st.set_page_config(layout="wide", page_title="Greek Registry Grid Extractor")
st.title("📜 Greek Registry Key-Value Grid Parser")

form_ids = [1, 2, 3]
labels_matrix = [
    ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ"],
    ["ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"]
]

if "extracted_values" not in st.session_state:
    st.session_state.extracted_values = {}

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

if uploaded_file and cred_file and st.button("🔍 Parse and Preview Forms"):
    np_image = np.array(image)
    h, w = np_image.shape[:2]
    form_h = h // 3
    left_w = w // 2
    pad = 5
    client = vision.ImageAnnotatorClient()

    for form_id in form_ids:
        y1, y2 = (form_id - 1) * form_h, form_id * form_h
        crop_np = np_image[y1:y2, :left_w].copy()
        preview_img = Image.fromarray(crop_np).convert("RGB")
        draw = ImageDraw.Draw(preview_img)

        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

        scale = 20
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,scale))
        horiz_lines = cv2.dilate(cv2.erode(binary, horiz_kernel), horiz_kernel)
        vert_lines = cv2.dilate(cv2.erode(binary, vert_kernel), vert_kernel)

        horiz_coords = np.where(np.sum(horiz_lines, axis=1) < horiz_lines.shape[1]*255)[0]
        vert_coords = np.where(np.sum(vert_lines, axis=0) < vert_lines.shape[0]*255)[0]

        rows = [horiz_coords[i] for i in range(0, len(horiz_coords), max(1, len(horiz_coords)//3))][:3]
        cols = [vert_coords[i] for i in range(0, len(vert_coords), max(1, len(vert_coords)//5))][:5]

        form_data = {}
        st.subheader(f"📄 Φόρμα {form_id}")

        for r in range(2):
            for c in range(4):
                field = labels_matrix[r][c]
                try:
                    y_start = max(0, rows[r] - pad)
                    y_end = rows[r+1] + pad if r+1 < len(rows) else crop_np.shape[0]
                    x_start = max(0, cols[c] - pad)
                    x_end = cols[c+1] + pad if c+1 < len(cols) else crop_np.shape[1]
                    cell = crop_np[y_start:y_end, x_start:x_end]

                    buffer = BytesIO()
                    Image.fromarray(cell).save(buffer, format="JPEG")
                    buffer.seek(0)
                    response = client.document_text_detection(image=vision.Image(content=buffer.getvalue()))
                    text = response.full_text_annotation.text.strip()
                    value = " ".join(text.split("\n")).strip()
                    form_data[field] = value

                    draw.rectangle([(x_start, y_start), (x_end, y_end)], outline="red", width=2)
                    draw.text((x_start + 5, y_start + 5), field, fill="blue")
                except:
                    form_data[field] = "—"

        # Save and reload image with annotations
        preview_buffer = BytesIO()
        preview_img.save(preview_buffer, format="PNG")
        preview_buffer.seek(0)
        preview_np = np.array(Image.open(preview_buffer))
        st.image(preview_np, caption=f"🖼️ Φόρμα {form_id} with Boxes & Labels", use_column_width=True)

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
