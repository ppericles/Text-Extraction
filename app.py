import streamlit as st
from PIL import Image, ImageDraw
import json
import os
from difflib import SequenceMatcher, get_close_matches
from google.cloud import vision
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO

from utils.image_utils import image_to_base64
from utils.ocr_utils import normalize, detect_header_regions, compute_form_bounds
from utils.layout_utils import get_form_bounding_box

st.set_page_config(layout="wide", page_title="Greek OCR Annotator")
st.title("🇬🇷 Greek OCR Annotator — JP2-Compatible Edition")

field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

form_ids = [1, 2, 3]

def convert_to_jpeg_bytes(pil_image):
    with BytesIO() as buffer:
        pil_image.save(buffer, format="JPEG")
        return buffer.getvalue()

# Initialize session state
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in form_ids}
if "ocr_blocks" not in st.session_state:
    st.session_state.ocr_blocks = []
if "click_points" not in st.session_state:
    st.session_state.click_points = []
if "extracted_values" not in st.session_state:
    st.session_state.extracted_values = {i: {} for i in form_ids}

view_mode = st.sidebar.radio("🧭 View Mode", ["Tagging", "Compare All Forms"])
form_num = st.sidebar.selectbox("📄 Select Φόρμα", form_ids)
selected_label = st.sidebar.selectbox("📝 Field Label", field_labels)

cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

layout_file = st.sidebar.file_uploader("📂 Import layout (.json)", type=["json"])
if layout_file:
    try:
        raw_layout = json.load(layout_file)
        st.session_state.form_layouts = {int(k): v for k, v in raw_layout.items()}
        st.sidebar.success("✅ Layout imported")
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")

uploaded_file = st.file_uploader(
    "📎 Upload scanned form",
    type=["jpg", "jpeg", "png", "jp2"],
    help="💡 Drag-and-drop may work better for .jp2 files due to browser MIME quirks"
)

if view_mode == "Tagging":
    if uploaded_file:
        filename = uploaded_file.name.lower()
        image = Image.open(uploaded_file).convert("RGB")
        field_boxes = st.session_state.form_layouts[form_num]

        st.markdown("### 👆 Click twice to define a field box")
        coords = streamlit_image_coordinates(image)
        if coords:
            st.session_state.click_points.append((coords["x"], coords["y"]))
            st.toast(f"Point {len(st.session_state.click_points)}: ({coords['x']}, {coords['y']})")

            if len(st.session_state.click_points) == 2:
                (x1, y1), (x2, y2) = st.session_state.click_points
                st.session_state.click_points = []

                nearby_texts = []
                for b in st.session_state.ocr_blocks:
                    cx, cy = b["center"]
                    if min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2):
                        nearby_texts.append(b["text"])
                candidate = " ".join(nearby_texts).upper()
                match = get_close_matches(candidate, field_labels, n=1)
                suggested = match[0] if match else selected_label
                score = round(SequenceMatcher(None, candidate, suggested).ratio() * 100, 2)

                field_boxes[suggested] = {
                    "x1": min(x1, x2), "y1": min(y1, y2),
                    "x2": max(x1, x2), "y2": max(y1, y2)
                }
                st.success(f"✅ Saved box for '{suggested}' (Confidence: {score}%) in Φόρμα {form_num}")

        if cred_file and st.button("🔍 Run OCR"):
            try:
                with st.spinner("Running OCR..."):
                    image_bytes = (
                        convert_to_jpeg_bytes(image)
                        if filename.endswith(".jp2")
                        else uploaded_file.getvalue()
                    )

                    client = vision.ImageAnnotatorClient()
                    vision_img = vision.Image(content=image_bytes)
                    response = client.document_text_detection(image=vision_img)
                    annotations = response.text_annotations

                    draw_img = image.copy()
                    draw = ImageDraw.Draw(draw_img)
                    blocks = []

                    for ann in annotations[1:]:
                        vertices = ann.bounding_poly.vertices
                        xs = [int(v.x) for v in vertices if v.x is not None]
                        ys = [int(v.y) for v in vertices if v.y is not None]
                        x1a, x2a = min(xs), max(xs)
                        y1a, y2a = min(ys), max(ys)
                        center = (sum(xs) / len(xs), sum(ys) / len(ys))
                        blocks.append({"text": ann.description, "center": center})
                        draw.rectangle([(x1a, y1a), (x2a, y2a)], outline="red", width=1)
                        draw.text((x1a, y1a - 10), ann.description, fill="blue")

                    for label in field_labels:
                        box = field_boxes.get(label)
                        if box:
                            draw.rectangle(
                                [(box["x1"], box["y1"]), (box["x2"], box["y2"])],
                                outline="green", width=3
                            )
                            draw.text((box["x1"], box["y1"] - 12), label, fill="green")

                    st.session_state.ocr_blocks = blocks

                st.markdown("### 🖼️ OCR Overlay")
                overlay_base64 = image_to_base64(draw_img)
                st.markdown(
                    f"""<div style='overflow-x:auto'><img src='data:image/png;base64,{overlay_base64}' /></div>""",
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"OCR failed: {e}")

        st.subheader(f"🧠 Extracted Fields — Φόρμα {form_num}")
        with st.expander(f"Fields for Φόρμα {form_num}", expanded=True):
            extracted = {}
            for label in field_labels:
                box = field_boxes.get(label)
                if box:
                    xmin, xmax = sorted([box["x1"], box["x2"]])
                    ymin, ymax = sorted([box["y1"], box["y2"]])
                    matches = [
                        b["text"] for b in st.session_state.ocr_blocks
                        if xmin <= b["center"][0] <= xmax and ymin <= b["center"][1] <= ymax
                    ]
                    val = " ".join(matches) if matches else ""
                    extracted[label] = val
                    st.text_input(label, val, key=f"{form_num}_{label}")
            st.session_state.extracted_values[form_num] = extracted

        st.download_button(
            label="💾 Export Layout",
            data=json.dumps(st.session_state.form_layouts, indent=2, ensure_ascii=False),
            file_name="form_layouts.json",
            mime="application/json"
        )

# === Compare Mode ===
if view_mode == "Compare All Forms":
    st.markdown("## 📊 Form Comparison Dashboard")
    cols = st.columns(len(form_ids))
    for idx, form_id in enumerate(form_ids):
        with cols[idx]:
            st.markdown(f"### Φόρμα {form_id}")
            field_boxes = st.session_state.form_layouts.get(form_id, {})
            extracted = st.session_state.extracted_values.get(form_id, {})
            tagged = list(field_boxes.keys())
            missing = [f for f in field_labels if f not in tagged]
            st.write(f"✅ Tagged: {len(tagged)} / {len(field_labels)}")
            if missing:
                st.write("❌ Missing:", ", ".join(missing))
            for label in field_labels:
                value = extracted.get(label, "(no value)")
                st.text_input(label, value, key=f"compare_{form_id}_{label}")
