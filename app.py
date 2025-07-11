import streamlit as st
from PIL import Image
import numpy as np
import json
import os
from google.cloud import vision
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Greek Handwriting OCR with Google Vision")
st.title("🇬🇷 Greek Handwriting Form Parser (Google Vision AI)")

field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# --- Session state setup
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in [1, 2, 3]}
if "click_stage" not in st.session_state:
    st.session_state.click_stage = "start"
if "ocr_blocks" not in st.session_state:
    st.session_state.ocr_blocks = []

# --- Sidebar controls
form_num = st.sidebar.selectbox("📄 Φόρμα", [1, 2, 3])
field_label = st.sidebar.selectbox("📝 Field Name", field_labels)

# --- Upload credentials file
cred_file = st.sidebar.file_uploader("🔐 Upload Google Vision credentials (JSON)", type=["json"])
if cred_file:
    cred_path = "credentials.json"
    with open(cred_path, "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
    st.sidebar.success("✅ Credentials loaded")

# --- Upload layout
layout_file = st.sidebar.file_uploader("📂 Import layout (.json)", type=["json"])
if layout_file:
    try:
        st.session_state.form_layouts = json.load(layout_file)
        st.sidebar.success("✅ Layout imported")
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")

# --- Upload form image
uploaded_file = st.file_uploader("📎 Upload scanned handwritten form", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="📷 Uploaded Image", use_container_width=True)
    st.caption("👆 Click to define bounding boxes for fields. Top-left then bottom-right.")

    coords = streamlit_image_coordinates(image, key="coord_click")
    if coords:
        x, y = coords["x"], coords["y"]
        field_boxes = st.session_state.form_layouts[form_num]
        if st.session_state.click_stage == "start":
            field_boxes[field_label] = {"x1": x, "y1": y}
            st.session_state.click_stage = "end"
            st.info(f"🟩 Top-left set for '{field_label}'. Click bottom-right.")
        else:
            field_boxes[field_label].update({"x2": x, "y2": y})
            st.session_state.click_stage = "start"
            st.success(f"✅ Box saved for '{field_label}' in Φόρμα {form_num}.")

    # --- Google Vision OCR
    if cred_file:
        client = vision.ImageAnnotatorClient()
        content = uploaded_file.getvalue()
        vision_img = vision.Image(content=content)
        response = client.document_text_detection(image=vision_img)
        text_annotations = response.text_annotations

        # Skip first item (entire text block)
        block_data = []
        for ann in text_annotations[1:]:
            bounds = [(v.x, v.y) for v in ann.bounding_poly.vertices]
            center = (np.mean([p[0] for p in bounds]), np.mean([p[1] for p in bounds]))
            block_data.append({"text": ann.description, "center": center})

        st.session_state.ocr_blocks = block_data

        # --- Display field values per form
        st.subheader("🧠 OCR Field Extraction")
        for i in [1, 2, 3]:
            st.markdown(f"### 📄 Φόρμα {i}")
            layout = st.session_state.form_layouts[i]
            for label in field_labels:
                box = layout.get(label)
                if box and all(k in box for k in ["x1", "y1", "x2", "y2"]):
                    xmin, xmax = sorted([box["x1"], box["x2"]])
                    ymin, ymax = sorted([box["y1"], box["y2"]])
                    match = next(
                        (b for b in st.session_state.ocr_blocks
                         if xmin <= b["center"][0] <= xmax and ymin <= b["center"][1] <= ymax),
                        None
                    )
                    val = match["text"] if match else "(no match)"
                    st.text_input(label, val, key=f"{i}_{label}")

# --- Export layout
st.download_button(
    label="💾 Export Layout as JSON",
    data=json.dumps(st.session_state.form_layouts, ensure_ascii=False, indent=2),
    file_name="form_layouts.json",
    mime="application/json"
)
