import streamlit as st
from PIL import Image, ImageDraw
import json
import os
from google.cloud import vision

from utils.ocr_utils import normalize, detect_header_regions, compute_form_bounds
from utils.image_utils import image_to_base64
from utils.layout_utils import get_form_bounding_box

st.set_page_config(layout="wide", page_title="Greek OCR Annotator")
st.title("🇬🇷 Greek OCR Annotator — Manual Tagging Edition")

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

form_num = st.sidebar.selectbox("📄 Select Form", [1, 2, 3])
selected_label = st.sidebar.selectbox("📝 Field Label", field_labels)

# Credentials
cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

# Layout importer
layout_file = st.sidebar.file_uploader("📂 Import layout (.json)", type=["json"])
if layout_file:
    try:
        st.session_state.form_layouts = json.load(layout_file)
        st.sidebar.success("✅ Layout imported")
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")

# Upload image
uploaded_file = st.file_uploader("📎 Upload scanned form", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    field_boxes = st.session_state.form_layouts[form_num]

    st.image(image, caption="Uploaded Form", use_column_width=True)

    st.markdown("### 📌 Define Bounding Box for Field")
    x1 = st.number_input("Top-left X", min_value=0, max_value=width, value=0)
    y1 = st.number_input("Top-left Y", min_value=0, max_value=height, value=0)
    x2 = st.number_input("Bottom-right X", min_value=0, max_value=width, value=0)
    y2 = st.number_input("Bottom-right Y", min_value=0, max_value=height, value=0)

    if st.button("📍 Save Box"):
        field_boxes[selected_label] = {
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2)
        }
        st.success(f"✅ Saved box for '{selected_label}' in Φόρμα {form_num}")

    # Run OCR
    if cred_file and st.button("🔍 Run OCR"):
        try:
            with st.spinner("Running OCR..."):
                client = vision.ImageAnnotatorClient()
                vision_img = vision.Image(content=uploaded_file.getvalue())
                response = client.document_text_detection(image=vision_img)
                annotations = response.text_annotations

                detect_header_regions(annotations, field_labels, field_boxes, debug=True)

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

                # Draw saved field boxes
                for label in field_labels:
                    box = field_boxes.get(label)
                    if box:
                        draw.rectangle(
                            [(box["x1"], box["y1"]), (box["x2"], box["y2"])],
                            outline="green", width=3
                        )
                        draw.text((box["x1"], box["y1"] - 12), label, fill="green")

                st.session_state.ocr_blocks = blocks

            st.markdown("### 🖼️ Overlay Image with OCR + Tags")
            overlay_base64 = image_to_base64(draw_img)
            st.markdown(
                f"""<div style='overflow-x:auto'><img src='data:image/png;base64,{overlay_base64}' /></div>""",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"OCR failed: {e}")

    st.subheader("🧠 Extracted Field Values")
    for label in field_labels:
        box = field_boxes.get(label)
        if box:
            xmin, xmax = sorted([box["x1"], box["x2"]])
            ymin, ymax = sorted([box["y1"], box["y2"]])
            matches = [
                b["text"] for b in st.session_state.ocr_blocks
                if xmin <= b["center"][0] <= xmax and ymin <= b["center"][1] <= ymax
            ]
            val = " ".join(matches) if matches else "(no match)"
            st.text_input(label, val, key=f"{form_num}_{label}")

    st.download_button(
        label="💾 Export Layout as JSON",
        data=json.dumps(st.session_state.form_layouts, indent=2, ensure_ascii=False),
        file_name="form_layouts.json",
        mime="application/json"
    )
