import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import json
import os
import base64
from io import BytesIO
from google.cloud import vision
from streamlit_image_coordinates import streamlit_image_coordinates
from unidecode import unidecode

def normalize(text):
    return unidecode(text.upper().strip())

def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

st.set_page_config(layout="wide", page_title="Greek OCR Annotator")
st.title("🇬🇷 Greek OCR Annotator — Scrollable Zoom + True Overlay")

# Field labels
field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# Session state
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in [1, 2, 3]}
if "click_stage" not in st.session_state:
    st.session_state.click_stage = "start"
if "ocr_blocks" not in st.session_state:
    st.session_state.ocr_blocks = []
if "auto_extracted_fields" not in st.session_state:
    st.session_state.auto_extracted_fields = {}

# Sidebar UI
form_num = st.sidebar.selectbox("📄 Φόρμα", [1, 2, 3])
field_label = st.sidebar.selectbox("📝 Field Name", field_labels)
zoom = st.sidebar.slider("🔍 Zoom", min_value=0.5, max_value=2.5, value=1.5, step=0.1)

cred_file = st.sidebar.file_uploader("🔐 Google credentials (JSON)", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

layout_file = st.sidebar.file_uploader("📂 Import layout (.json)", type=["json"])
if layout_file:
    try:
        st.session_state.form_layouts = json.load(layout_file)
        st.sidebar.success("✅ Layout imported")
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")

# Image upload
uploaded_file = st.file_uploader("📎 Upload scanned form", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    scaled_image = image.resize((int(image.width * zoom), int(image.height * zoom)))
    field_boxes = st.session_state.form_layouts[form_num]

    # Tagging interface with click coordinates
    st.markdown("### 🖱️ Tagging Image (Click top-left then bottom-right)")
    coords = streamlit_image_coordinates(
        scaled_image, 
        key="coord_click",
        height=min(800, scaled_image.height)
    )

    if coords:
        # Only process clicks if they're within the image bounds
        if 0 <= coords["x"] < scaled_image.width and 0 <= coords["y"] < scaled_image.height:
            x, y = coords["x"], coords["y"]
            if st.session_state.click_stage == "start":
                field_boxes[field_label] = {"x1": x, "y1": y}
                st.session_state.click_stage = "end"
                st.info(f"🟩 Top-left set for '{field_label}'. Click bottom-right.")
            else:
                field_boxes[field_label].update({"x2": x, "y2": y})
                st.session_state.click_stage = "start"
                st.success(f"✅ Box saved for '{field_label}' in Φόρμα {form_num}.")
        else:
            st.warning("⚠️ Please click inside the image area")

    if cred_file:
        client = vision.ImageAnnotatorClient()
        vision_img = vision.Image(content=uploaded_file.getvalue())
        response = client.document_text_detection(image=vision_img)
        annotations = response.text_annotations

        draw_img = scaled_image.copy()
        draw = ImageDraw.Draw(draw_img)
        blocks = []

        for ann in annotations[1:]:
            vertices = ann.bounding_poly.vertices
            xs = [int(v.x * zoom) for v in vertices]
            ys = [int(v.y * zoom) for v in vertices]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            center = (np.mean(xs), np.mean(ys))
            blocks.append({"text": ann.description, "center": center})
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
            draw.text((x1, y1 - 10), ann.description, fill="blue")

        layout = st.session_state.form_layouts[form_num]
        for label, box in layout.items():
            if all(k in box for k in ("x1", "y1", "x2", "y2")):
                x1, y1 = box["x1"], box["y1"]
                x2, y2 = box["x2"], box["y2"]
                draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
                draw.text((x1, y1 - 10), label, fill="green")

        st.session_state.ocr_blocks = blocks

        # Tagged OCR overlay with horizontal scroll
        st.markdown("### 📌 Tagged OCR Overlay")
        overlay_base64 = image_to_base64(draw_img)
        st.markdown(
            f"""
            <div style='overflow-x:auto; border:1px solid #ccc; padding:10px; white-space:nowrap;'>
                <img src='data:image/png;base64,{overlay_base64}' style='max-height: 800px; width: auto; display:block;' />
            </div>
            """,
            unsafe_allow_html=True
        )

        # Manual extraction
        st.subheader("🧠 Extracted Field Values")
        for i in [1, 2, 3]:
            st.markdown(f"### 📄 Φόρμα {i}")
            layout = st.session_state.form_layouts[i]
            for label in field_labels:
                box = layout.get(label)
                if box and all(k in box for k in ("x1", "y1", "x2", "y2")):
                    xmin, xmax = sorted([box["x1"], box["x2"]])
                    ymin, ymax = sorted([box["y1"], box["y2"]])
                    matches = [
                        b["text"] for b in st.session_state.ocr_blocks
                        if xmin <= b["center"][0] <= xmax and ymin <= b["center"][1] <= ymax
                    ]
                    val = " ".join(matches) if matches else "(no match)"
                    st.text_input(label, val, key=f"{i}_{label}")

        # Auto extraction
        st.header("🪄 Auto-Extracted Fields")
        if st.button("🪄 Auto-Extract from OCR"):
            found = {}
            normalized_labels = {normalize(lbl): lbl for lbl in field_labels}
            for idx, block in enumerate(st.session_state.ocr_blocks):
                txt = normalize(block["text"])
                if txt in normalized_labels:
                    ref_x, ref_y = block["center"]
                    neighbor = None
                    min_dist = float("inf")
                    for other in st.session_state.ocr_blocks[idx+1:]:
                        dx = other["center"][0] - ref_x
                        dy = other["center"][1] - ref_y
                        if dx >= 0 and dy >= 0:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < min_dist:
                                min_dist = dist
                                neighbor = other
                    if neighbor:
                        label = normalized_labels[txt]
                        found[label] = neighbor["text"]
            st.session_state.auto_extracted_fields = found

        if st.session_state.auto_extracted_fields:
            st.subheader("🧾 Predicted Field Mapping")
            st.json(st.session_state.auto_extracted_fields)

# Export layout
st.download_button(
    label="💾 Export Layout as JSON",
    data=json.dumps(st.session_state.form_layouts, ensure_ascii=False, indent=2),
    file_name="form_layouts.json",
    mime="application/json"
)
