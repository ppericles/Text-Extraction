import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
from streamlit_image_coordinates import streamlit_image_coordinates
import json

# --- Google Cloud Vision client ---
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = vision.ImageAnnotatorClient(credentials=credentials)

field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# --- Session State ---
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in [1, 2, 3]}
if "click_stage" not in st.session_state:
    st.session_state.click_stage = "start"
if "ocr_blocks" not in st.session_state:
    st.session_state.ocr_blocks = []

st.set_page_config(layout="wide", page_title="Greek Form Bounding Box Tagger")
st.title("📐 Two-Click Field Annotation with Live OCR Matching")

# --- Layout Import
uploaded_layout = st.file_uploader("📂 Import Layout (.json)", type=["json"])
if uploaded_layout:
    try:
        st.session_state.form_layouts = json.load(uploaded_layout)
        st.success("✅ Layout imported successfully")
    except Exception as e:
        st.error(f"Layout import failed: {e}")

# --- Sidebar Field Selector
form_num = st.sidebar.selectbox("📄 Select Form", [1, 2, 3])
field_label = st.sidebar.selectbox("📝 Field Name", field_labels)

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
    st.caption("👆 Click top-left, then bottom-right to define a bounding box.")

    # --- OCR Block Extraction
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

    st.session_state.ocr_blocks = blocks  # cache

    # --- Two-Click Bounding Box Logic
    coords = streamlit_image_coordinates(img, key="click_box")
    if coords:
        x, y = coords["x"], coords["y"]
        field_boxes = st.session_state.form_layouts[form_num]

        if st.session_state.click_stage == "start":
            field_boxes[field_label] = {"x1": x, "y1": y}
            st.session_state.click_stage = "end"
            st.info(f"🟩 Top-left corner set for '{field_label}'. Now click bottom-right.")
        else:
            if field_label not in field_boxes:
                field_boxes[field_label] = {}
            field_boxes[field_label].update({"x2": x, "y2": y})
            st.session_state.click_stage = "start"
            st.success(f"✅ Box saved for '{field_label}' in Φόρμα {form_num}.")

            # --- Auto OCR Match Preview
            x1, x2 = field_boxes[field_label]["x1"], field_boxes[field_label]["x2"]
            y1, y2 = field_boxes[field_label]["y1"], field_boxes[field_label]["y2"]
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            match = next(
                (b for b in st.session_state.ocr_blocks if xmin <= b["x"] <= xmax and ymin <= b["y"] <= ymax),
                None
            )
            preview = match["text"] if match else "(no match)"
            st.info(f"🔍 Field Preview for '{field_label}': {preview}")

    # --- Full Field Values Display
    st.markdown("## 🧠 Extracted Field Values")
    for i in [1, 2, 3]:
        st.subheader(f"📄 Φόρμα {i}")
        layout = st.session_state.form_layouts.get(i, {})
        for label in field_labels:
            box = layout.get(label)
            if box and all(k in box for k in ["x1", "y1", "x2", "y2"]):
                xmin, xmax = sorted([box["x1"], box["x2"]])
                ymin, ymax = sorted([box["y1"], box["y2"]])
                match = next(
                    (b for b in st.session_state.ocr_blocks if xmin <= b["x"] <= xmax and ymin <= b["y"] <= ymax),
                    None
                )
                val = match["text"] if match else ""
                st.text_input(label, val, key=f"{i}_{label}")

# --- Layout Export
st.download_button(
    label="💾 Download Layout as JSON",
    data=json.dumps(st.session_state.form_layouts, ensure_ascii=False, indent=2),
    file_name="form_field_boxes.json",
    mime="application/json"
)
