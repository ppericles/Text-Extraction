import streamlit as st
from PIL import Image, ImageDraw
import os, json
from io import BytesIO
import numpy as np
import pandas as pd
import cv2
from google.cloud import vision
from google.cloud.vision_v1 import types

st.set_page_config(layout="wide", page_title="Greek Registry Visual OCR")
st.title("📜 Greek Registry Parser with Box Intelligence")

form_ids = [1, 2, 3]
labels_matrix = [
    ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ"],
    ["ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"]
]
field_categories = {
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": "admin", "ΚΑΤΟΙΚΙΑ": "admin",
    "ΕΠΩΝΥΜΟ": "identity", "ΚΥΡΙΟΝ ΟΝΟΜΑ": "identity", "ΟΝΟΜΑ ΠΑΤΡΟΣ": "identity",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ": "identity",
    "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ": "origin", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ": "origin"
}
color_modes = {"Confidence": True, "Field Type": False, "Classic": None}
mode = st.sidebar.selectbox("🎨 Color Mode", list(color_modes.keys()))

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

if uploaded_file and cred_file and st.button("🔍 Run OCR with Smart Boxes"):
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

        form_data = {}
        for idx, box in enumerate(boxes[:8]):
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
            confidence = 0.0
            if response.text_annotations and len(response.text_annotations) > 0:
                confidence = max((a.confidence for a in response.text_annotations if hasattr(a, 'confidence')), default=0.0)
                confidence *= 100
            field = labels_matrix[idx // 4][idx % 4]
            form_data[field] = value

            # Determine box color
            if color_modes[mode] is True:
                if confidence >= 90:
                    box_color = "green"
                elif confidence >= 70:
                    box_color = "yellow"
                else:
                    box_color = "red"
            elif color_modes[mode] is False:
                cat = field_categories.get(field, "identity")
                box_color = {
                    "identity": "blue",
                    "origin": "purple",
                    "admin": "green"
                }.get(cat, "gray")
            else:
                box_color = "red"

            draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=2)
            draw.text((x1 + 4, y1 + 4), f"{field} ({confidence:.1f}%)", fill="blue")

        buffer = BytesIO()
        preview_img.save(buffer, format="PNG")
        buffer.seek(0)
        annotated = Image.open(buffer)
        st.image(np.array(annotated), caption=f"🖼️ Φόρμα {form_id} — {mode} Mode", use_column_width=True)

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
