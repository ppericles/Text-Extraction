import streamlit as st
from PIL import Image, ImageDraw
import json
import os
import numpy as np
from google.cloud import vision
from streamlit_drawable_canvas import st_canvas

from utils.ocr_utils import normalize, detect_header_regions, compute_form_bounds
from utils.image_utils import image_to_base64
from utils.layout_utils import get_form_bounding_box

st.set_page_config(layout="wide", page_title="Greek OCR Annotator")
st.title("🇬🇷 Greek OCR Annotator — Drag-to-Tag Edition")

field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# Initialize session state
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in [1, 2, 3]}
if "ocr_blocks" not in st.session_state:
    st.session_state.ocr_blocks = []
if "auto_extracted_fields" not in st.session_state:
    st.session_state.auto_extracted_fields = {}

form_num = st.sidebar.selectbox("📄 Φόρμα", [1, 2, 3])
selected_label = st.sidebar.selectbox("📝 Select Field Label", field_labels)

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
background_image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    background_image = np.array(image).astype(np.uint8)
    field_boxes = st.session_state.form_layouts[form_num]

    st.markdown("### 🖱️ Drag to Tag Field Regions")
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",
        stroke_width=2,
        stroke_color="green",
        background_image=background_image if background_image is not None else None,
        height=height,
        width=width,
        drawing_mode="rect",
        key="canvas"
    )

    if canvas_result.json_data and canvas_result.json_data["objects"]:
        latest = canvas_result.json_data["objects"][-1]
        x1 = int(latest["left"])
        y1 = int(latest
