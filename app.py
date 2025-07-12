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

# Initialize session state
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in [1, 2, 3]}
if "click_stage" not in st.session_state:
    st.session_state.click_stage = "start"
if "ocr_blocks" not in st.session_state:
    st.session_state.ocr_blocks = []
if "auto_extracted_fields" not in st.session_state:
    st.session_state.auto_extracted_fields = {}
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""
if "current_zoom" not in st.session_state:
    st.session_state.current_zoom = 1.0
if "scaled_image" not in st.session_state:
    st.session_state.scaled_image = None

st.set_page_config(layout="wide", page_title="Greek OCR Annotator")
st.title("🇬🇷 Greek OCR Annotator — Scrollable Zoom + True Overlay")

# Field labels
field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# Sidebar UI
form_num = st.sidebar.selectbox("📄 Φόρμα", [1, 2, 3])
field_label = st.sidebar.selectbox("📝 Field Name", field_labels)

# Zoom slider with callback
def update_zoom():
    if st.session_state.original_image:
        st.session_state.current_zoom = st.session_state.zoom_slider
        # Create scaled image with proper parentheses
        st.session_state.scaled_image = st.session_state.original_image.resize(
            (int(st.session_state.original_image.width * st.session_state.current_zoom),
            int(st.session_state.original_image.height * st.session_state.current_zoom)
        )

zoom = st.sidebar.slider(
    "🔍 Zoom", 
    min_value=0.5, 
    max_value=2.5, 
    value=st.session_state.current_zoom, 
    step=0.1,
    key="zoom_slider",
    on_change=update_zoom
)

# [Rest of your code remains exactly the same...]

# Image upload handling
uploaded_file = st.file_uploader("📎 Upload scanned form", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Only process if we have a new file
    if uploaded_file.name != st.session_state.uploaded_file_name:
        try:
            st.session_state.original_image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.ocr_blocks = []
            st.session_state.auto_extracted_fields = {}
            # Create initial scaled image with proper parentheses
            st.session_state.scaled_image = st.session_state.original_image.resize(
                (int(st.session_state.original_image.width * st.session_state.current_zoom),
                int(st.session_state.original_image.height * st.session_state.current_zoom)
            )
            st.success("🔄 Image loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load image: {e}")
            st.stop()
    
    if st.session_state.scaled_image:
        scaled_image = st.session_state.scaled_image
        field_boxes = st.session_state.form_layouts[form_num]

        # [Rest of your existing code...]

# Export layout
st.download_button(
    label="💾 Export Layout as JSON",
    data=json.dumps(st.session_state.form_layouts, ensure_ascii=False, indent=2),
    file_name="form_layouts.json",
    mime="application/json"
)
