import streamlit as st
from PIL import Image, ImageDraw
import json
import os
from google.cloud import vision
from streamlit_image_coordinates import streamlit_image_coordinates

from utils.ocr_utils import normalize, detect_header_regions
from utils.image_utils import image_to_base64
from utils.layout_utils import get_form_bounding_box
from utils.tagging_utils import handle_click

st.set_page_config(layout="wide", page_title="Greek OCR Annotator")
st.title("🇬🇷 Greek OCR Annotator — Auto Header Detection + OCR Overlay")

field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in [1, 2, 3]}
if "click_stage" not in st.session_state:
    st.session_state.click_stage = "start"
if "ocr_blocks" not in st.session_state:
    st.session_state.ocr_blocks = []
if "auto_extracted_fields" not in st.session_state:
    st.session_state.auto_extracted_fields = {}

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

    st.markdown("### 🖱️ Tagging Image")
    coords = streamlit_image_coordinates(image, key="coord_click", height=height)

    if coords:
        result = handle_click(coords, width, height, field_label, field_boxes, st.session_state)
        if result == "outside":
            st.warning("⚠️ Click was outside image bounds. Try again.")
        elif result == "start":
            st.info(f"🟩 Top-left set for '{field_label}'. Click bottom-right.")
        else:
            st.success(f"✅ Box saved for '{field_label}' in Φόρμα {form_num}.")

    if cred_file:
        try:
            with st.spinner("🔍 Running OCR..."):
                client = vision.ImageAnnotatorClient()
                vision_img = vision.Image(content=uploaded_file.getvalue())
                response = client.document_text_detection(image=vision_img)
                annotations = response.text_annotations

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

                for i in [1, 2, 3]:
                    form_boxes = st.session_state.form_layouts[i]
                    bounds = get_form_bounding_box(form_boxes)
                    if bounds:
                        x_min, y_min, x_max, y_max = bounds
                        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="green", width=3)
                        draw.text((x_min, y_min - 30), f"Φόρμα {i}", fill="green")

                st.session_state.ocr_blocks = blocks

            st.markdown("### 📌 Tagged OCR Overlay")
            overlay_base64 = image_to_base64(draw_img)
            st.markdown(
                f"""
                <div style='overflow-x:auto; border:1px solid #ccc; padding:10px; white-space:nowrap;'>
                    <img src='data:image/png;base64,{overlay_base64}' style='max-height:800px; width:auto; display:block;' />
                </div>
                """,
                unsafe_allow_html=True
            )

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

        except Exception as e:
            st.error(f"OCR processing failed: {e}")

    st.download_button(
        label="💾 Export Layout as JSON",
        data=json.dumps(st.session_state.form_layouts, ensure_ascii=False, indent=2),
        file_name="form_layouts.json",
        mime="application/json"
    )
