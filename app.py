import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
from streamlit_image_coordinates import streamlit_image_coordinates
import json

# --- Google Cloud Vision client ---
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = vision.ImageAnnotatorClient(credentials=credentials)

# --- Define Fields
field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# --- Session Setup
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in [1, 2, 3]}
if "click_stage" not in st.session_state:
    st.session_state.click_stage = "start"

# --- UI Config
st.set_page_config(layout="wide", page_title="Greek Form Field Calibrator")
st.title("📐 Field Box Tagger — Two-Click Interface")

# --- Import Existing Layout
uploaded_layout = st.file_uploader("📂 Import Layout (.json)", type=["json"])
if uploaded_layout:
    try:
        st.session_state.form_layouts = json.load(uploaded_layout)
        st.success("✅ Layout imported")
    except Exception as e:
        st.error(f"Error loading layout: {e}")

# --- Sidebar Config
form_num = st.sidebar.selectbox("📄 Select Form", [1, 2, 3])
field_label = st.sidebar.selectbox("📝 Select Field", field_labels)

# --- Image Upload
uploaded_file = st.file_uploader("📎 Upload scanned Greek form", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > 1800:
        img = img.resize((min(img.width, 1800), min(img.height, 1800)))
    img_width, img_height = img.size

    st.image(img, caption="📷 Uploaded Form", use_container_width=True)
    st.caption("👆 Click to define bounding box: top-left first, then bottom-right.")

    # --- Two-Click Box Capture
    coords = streamlit_image_coordinates(img, key="click_box")
    if coords:
        x, y = coords["x"], coords["y"]
        field_boxes = st.session_state.form_layouts[form_num]
        if st.session_state.click_stage == "start":
            field_boxes[field_label] = {"x1": x, "y1": y}
            st.session_state.click_stage = "end"
            st.info(f"🟩 Top-left corner set for '{field_label}'. Now click bottom-right.")
        else:
            field_boxes[field_label].update({"x2": x, "y2": y})
            st.session_state.click_stage = "start"
            st.success(f"✅ Box saved for '{field_label}' in Φόρμα {form_num}.")

    # --- OCR
    uploaded_file.seek(0)
    try:
        image_proto = vision.Image(content=uploaded_file.read())
        response = client.document_text_detection(image=image_proto)
    except Exception as e:
        st.error(f"OCR failed: {e}")
        st.stop()

    blocks = []
    for block in response.full_text_annotation.pages[0].blocks:
        text = "".join(
            symbol.text
            for para in block.paragraphs
            for word in para.words
            for symbol in word.symbols
        ).strip()
        if text:
            x = sum(v.x for v in block.bounding_box.vertices) / 4
            y = sum(v.y for v in block.bounding_box.vertices) / 4
            blocks.append({"text": text, "x": x, "y": y})

    # --- Field Matching
    st.markdown("## 🧠 Extracted Field Values")
    for i in [1, 2, 3]:
        st.subheader(f"📄 Φόρμα {i}")
        layout = st.session_state.form_layouts.get(i, {})
        for label in field_labels:
            box = layout.get(label)
            if box and all(k in box for k in ["x1", "y1", "x2", "y2"]):
                x_min, x_max = sorted([box["x1"], box["x2"]])
                y_min, y_max = sorted([box["y1"], box["y2"]])
                match = next(
                    (b for b in blocks if x_min <= b["x"] <= x_max and y_min <= b["y"] <= y_max),
                    None
                )
                val = match["text"] if match else "(no match)"
                st.text_input(label, val, key=f"{i}_{label}")

# --- Export JSON Layout
st.download_button(
    label="💾 Export Layout as JSON",
    data=json.dumps(st.session_state.form_layouts, ensure_ascii=False, indent=2),
    file_name="form_field_boxes.json",
    mime="application/json"
)
