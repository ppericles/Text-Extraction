import streamlit as st
from PIL import Image, ImageDraw
import json
import os
from google.cloud import vision
from streamlit_image_coordinates import streamlit_image_coordinates

from utils.ocr_utils import normalize, detect_header_regions, compute_form_bounds
from utils.image_utils import image_to_base64
from utils.layout_utils import get_form_bounding_box
from utils.tagging_utils import handle_click

st.set_page_config(layout="wide", page_title="Greek OCR Annotator")
st.title("🇬🇷 Greek OCR Annotator — Modular Edition")

field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# Initialize session state
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in [1, 2, 3]}
if "click_stage" not in st.session_state:
    st.session_state.click_stage = "start"
if "ocr_blocks" not in st.session_state:
    st.session_state.ocr_blocks = []

# Sidebar controls
form_num = st.sidebar.selectbox("📄 Φόρμα", [1, 2, 3])
field_label = st.sidebar.selectbox("📝 Field Name", field_labels)

cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
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

uploaded_file = st.file_uploader("📎 Upload scanned form", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    field_boxes = st.session_state.form_layouts[form_num]

    st.markdown("### 🖱️ Manual Tagging")
    coords = streamlit_image_coordinates(image, key="coord_click", height=height)

    if coords:
        result = handle_click(coords, width, height, field_label, field_boxes, st.session_state)
        if result == "outside":
            st.warning("⚠️ Click outside image bounds")
        elif result == "start":
            st.info(f"🟩 Top-left set for '{field_label}'")
        else:
            st.success(f"✅ Box saved for '{field_label}'")

    if cred_file:
        try:
            with st.spinner("🔍 Running OCR..."):
                client = vision.ImageAnnotatorClient()
                vision_img = vision.Image(content=uploaded_file.getvalue())
                response = client.document_text_detection(image=vision_img)
                annotations = response.text_annotations

                # Auto-detect header fields
                detect_header_regions(annotations, field_labels, field_boxes)

                draw_img = image.copy()
                draw = ImageDraw.Draw(draw_img)
                blocks = []

                for ann in annotations[1:]:
                    vertices = ann.bounding_poly.vertices
                    xs = [int(v.x) for v in vertices if v.x is not None]
                    ys = [int(v.y) for v in vertices if v.y is not None]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    center = (sum(xs) / len(xs), sum(ys) / len(ys))
                    blocks.append({"text": ann.description, "center": center})
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=1)
                    draw.text((x1, y1 - 10), ann.description, fill="blue")

                # Draw header boxes
                for label in field_labels:
                    box = field_boxes.get(label)
                    if box and all(k in box for k in ("x1", "y1", "x2", "y2")):
                        x1, y1 = box["x1"], box["y1"]
                        x2, y2 = box["x2"], box["y2"]
                        draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
                        draw.text((x1, y1 - 12), label, fill="green")

                # Draw full form bounding box
                form_bounds = compute_form_bounds(field_boxes)
                if form_bounds:
                    x_min, y_min, x_max, y_max = form_bounds
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="green", width=4)
                    draw.text((x_min, y_min - 30), f"Φόρμα {form_num}", fill="green")

                st.session_state.ocr_blocks = blocks

            st.markdown("### 📌 Tagged Overlay")
            overlay_base64 = image_to_base64(draw_img)
            st.markdown(
                f"""
                <div style='border:1px solid #ccc; overflow-x:auto; padding:10px'>
                    <img src='data:image/png;base64,{overlay_base64}' style='max-height:800px' />
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"OCR failed: {e}")

    st.download_button(
        label="💾 Export Layout as JSON",
        data=json.dumps(st.session_state.form_layouts, indent=2, ensure_ascii=False),
        file_name="form_layouts.json",
        mime="application/json"
    )
