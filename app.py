import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
from streamlit_image_coordinates import streamlit_image_coordinates
import json

# --- Vision Setup ---
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = vision.ImageAnnotatorClient(credentials=credentials)

# --- Field Definitions ---
field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# --- Session State Layout ---
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in [1, 2, 3]}
if "click_stage" not in st.session_state:
    st.session_state.click_stage = "start"
if "selected_field" not in st.session_state:
    st.session_state.selected_field = field_labels[0]
if "click_form" not in st.session_state:
    st.session_state.click_form = 1

# --- UI Setup ---
st.set_page_config(layout="wide", page_title="Greek OCR Bounding Box Calibrator")
st.title("📐 Greek Form Parser — Two-Click Bounding Box Calibrator")

uploaded_layout = st.file_uploader("📂 Import Layout from JSON", type=["json"])
if uploaded_layout:
    try:
        st.session_state.form_layouts = json.load(uploaded_layout)
        st.success("✅ Layout imported")
    except Exception as e:
        st.error(f"Layout import failed: {e}")

# --- Sidebar UI ---
st.sidebar.markdown("## 🖱️ Select Field and Form")
st.sidebar.selectbox("📄 Φόρμα", [1, 2, 3], key="click_form")
st.sidebar.selectbox("📝 Field Name", field_labels, key="selected_field")

# --- Image Upload ---
uploaded_file = st.file_uploader("📎 Upload scanned Greek form", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > 1800:
        img = img.resize((min(img.width, 1800), min(img.height, 1800)))
    img_width, img_height = img.size
    st.image(img, caption="📷 Uploaded Form", use_column_width=True)

    image_coordinates = streamlit_image_coordinates("🖱️ Click to define field box", img, key="bbox_click")
    if image_coordinates:
        x, y = image_coordinates["x"], image_coordinates["y"]
        current_field = st.session_state.selected_field
        current_form = st.session_state.click_form

        if st.session_state.click_stage == "start":
            st.session_state.form_layouts[current_form][current_field] = {"x1": x, "y1": y}
            st.session_state.click_stage = "end"
            st.info("✅ Top-left corner captured. Click bottom-right next.")
        else:
            st.session_state.form_layouts[current_form][current_field].update({"x2": x, "y2": y})
            st.session_state.click_stage = "start"
            st.success("✅ Bounding box completed for field.")

    # --- OCR
    uploaded_file.seek(0)
    image_proto = vision.Image(content=uploaded_file.read())
    try:
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
    st.markdown("## 🧠 OCR Field Values")
    for form_num in [1, 2, 3]:
        st.subheader(f"📄 Φόρμα {form_num}")
        layout = st.session_state.form_layouts.get(form_num, {})
        for label in field_labels:
            bbox = layout.get(label)
            if bbox and all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                x_min = min(bbox["x1"], bbox["x2"])
                x_max = max(bbox["x1"], bbox["x2"])
                y_min = min(bbox["y1"], bbox["y2"])
                y_max = max(bbox["y1"], bbox["y2"])

                match = next(
                    (b for b in blocks if x_min <= b["x"] <= x_max and y_min <= b["y"] <= y_max),
                    None
                )
                val = match["text"] if match else "(no match)"
                st.text_input(label, val, key=f"{form_num}_{label}")

# --- Export
st.download_button(
    label="💾 Export Layout as JSON",
    data=json.dumps(st.session_state.form_layouts, ensure_ascii=False, indent=2),
    file_name="bounding_boxes_layout.json",
    mime="application/json"
)
