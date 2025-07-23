# =============================================================================
# FILE: app.py
# VERSION: 3.9.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Streamlit Registry Parser with canvas editing,
#              manual OCR, layout overlays, and export.
# =============================================================================

import streamlit as st
from PIL import Image
import os, json, tempfile
from streamlit_drawable_canvas import st_canvas
from cryptography.fernet import Fernet

from utils_image import (
    resize_for_preview,
    trim_whitespace,
    adaptive_trim_whitespace,
    draw_column_breaks,
    draw_row_breaks
)
from utils_layout import (
    draw_layout_overlay,
    validate_layout_for_preview,
    draw_layout_overlay_preview
)
from utils_parser import process_single_form

# === Helper: Update Boxes Safely ===
def update_boxes_if_changed(file_key: str, new_boxes: list):
    old_boxes = st.session_state.saved_boxes.get(file_key, [])
    if new_boxes != old_boxes:
        st.session_state.saved_boxes[file_key] = new_boxes
        return True
    return False

# === Helper: Convert Boxes for Canvas ===
def convert_boxes_to_canvas_objects(boxes, scale=1.0):
    objects = []
    for box in boxes:
        x1, y1, x2, y2 = box
        left = x1 * scale
        top = y1 * scale
        width = (x2 - x1) * scale
        height = (y2 - y1) * scale
        objects.append({
            "type": "rect",
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "fill": "rgba(255, 0, 0, 0.3)",
            "stroke": "red",
            "strokeWidth": 2
        })
    return {"objects": objects}

# === UI Setup ===
st.set_page_config(page_title="📄 Registry Parser", layout="wide")
st.title("📄 Registry Form Parser")

# === Credential Loading ===
st.sidebar.markdown("### 🔐 Load Google Credentials")
cred_file = st.sidebar.file_uploader("Upload JSON credentials", type=["json"])
if cred_file:
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_path.write(cred_file.read())
    temp_path.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path.name
    st.sidebar.success("✅ Credentials loaded.")
else:
    st.sidebar.warning("⚠️ OCR disabled — upload service account JSON.")

# === OCR Engine Selection ===
st.sidebar.markdown("### 🧠 OCR Engine")
ocr_engine = st.sidebar.radio("Choose OCR Engine", ["Vision API", "Document AI"])
use_docai = ocr_engine == "Document AI"

# === Image Settings ===
st.sidebar.markdown("### 🖼️ Image Settings")
use_adaptive_trim = st.sidebar.checkbox("Use Adaptive Trimming", value=True)

# === File Upload ===
uploaded_files = st.file_uploader("📤 Upload Registry Scans", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# === Session Initialization ===
if "saved_boxes" not in st.session_state:
    st.session_state.saved_boxes = {}
if "parsed_forms" not in st.session_state:
    st.session_state.parsed_forms = {}
# === Main Processing Loop ===
if uploaded_files:
    for file in uploaded_files:
        st.header(f"📄 `{file.name}` — Select Forms")

        try:
            image_raw = Image.open(file).convert("RGB")
            processed = adaptive_trim_whitespace(image_raw.copy()) if use_adaptive_trim else trim_whitespace(image_raw.copy())
            preview_img = resize_for_preview(processed)
            st.image(preview_img, caption="Preview Image", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Failed to process or preview image: {e}")
            continue

        # === Canvas Drawing Mode ===
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🖊️ Draw New Boxes", key=f"btn_draw_{file.name}"):
                st.session_state[f"drawing_mode_{file.name}"] = "rect"
        with col2:
            if st.button("🔧 Resize / Move Boxes", key=f"btn_edit_{file.name}"):
                st.session_state[f"drawing_mode_{file.name}"] = "transform"

        drawing_mode = st.session_state.get(f"drawing_mode_{file.name}", "rect")
        scale_factor = 1.0 / (processed.width / preview_img.width)
        canvas_json = convert_boxes_to_canvas_objects(
            st.session_state.saved_boxes.get(file.name, []),
            scale=scale_factor
        )

        canvas_result = st_canvas(
            background_image=preview_img,
            initial_drawing=canvas_json,
            drawing_mode=drawing_mode,
            display_toolbar=True,
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            height=preview_img.height,
            width=preview_img.width,
            update_streamlit=True,
            key=f"canvas_{file.name}"
        )

        updated_boxes = []
        if canvas_result and canvas_result.json_data:
            scale_x = processed.width / preview_img.width
            scale_y = processed.height / preview_img.height
            for obj in canvas_result.json_data.get("objects", []):
                try:
                    x1 = int(obj["left"] * scale_x)
                    y1 = int(obj["top"] * scale_y)
                    x2 = int((obj["left"] + obj["width"]) * scale_x)
                    y2 = int((obj["top"] + obj["height"]) * scale_y)
                    updated_boxes.append((x1, y1, x2, y2))
                except Exception as e:
                    st.warning(f"⚠️ Could not convert box: {e}")

            boxes_changed = update_boxes_if_changed(file.name, updated_boxes)

        form_boxes = st.session_state.saved_boxes.get(file.name, [])
        parsed_results = []

        for i, box in enumerate(form_boxes):
            x1, y1, x2, y2 = box
            form_crop = processed.crop((x1, y1, x2, y2))
            st.subheader(f"🧾 Form {i+1}")
            st.image(resize_for_preview(form_crop), caption="📄 Cropped Form", use_column_width=True)

            layout = {
                "master_ratio": 0.5,
                "group_a_box": [0.0, 0.0, 0.2, 1.0],
                "group_b_box": [0.2, 0.0, 1.0, 0.5],
                "detail_box": [0.0, 0.0, 1.0, 1.0],
                "auto_detect": True
            }

            st.image(resize_for_preview(draw_layout_overlay(form_crop, layout)), caption="🔍 Layout Overlay", use_column_width=True)

            if st.checkbox("🔬 Show layout preview with dummy box detection", key=f"preview_{file.name}_{i}"):
                layout_preview = {
                    "group_a": {"box": layout.get("group_a_box")},
                    "group_b": {"box": layout.get("group_b_box")},
                    "detail_zone": {"box": layout.get("detail_box")}
                }
                layout_preview = validate_layout_for_preview(layout_preview, form_crop.width, form_crop.height)
                preview_image = draw_layout_overlay_preview(form_crop.copy(), layout_preview)
                st.image(resize_for_preview(preview_image), caption="🖍️ Layout Preview (Validated)", use_column_width=True)

            if st.button(f"🔍 Run OCR for Form {i+1}", key=f"ocr_btn_{file.name}_{i}"):
                config = {}  # Replace with docai_config if needed
                result = process_single_form(form_crop, i, config, layout)
                parsed_results.append(result)

                st.image(resize_for_preview(draw_column_breaks(result["table_crop"], result["column_breaks"])), caption="📊 Column Breaks", use_column_width=True)
                st.image(resize_for_preview(draw_row_breaks(result["table_crop"], rows=10, header=True)), caption="📏 Row Breaks", use_column_width=True)

        st.session_state.parsed_forms[file.name] = parsed_results

        st.markdown("## 📦 Export All Forms")
        if st.button("📤 Export All Parsed Data", key=f"export_all_{file.name}"):
            all_data = {
                f"form_{i+1}": {
                    "group_a": r["group_a"],
                    "group_b": r["group_b"],
                    "table_rows": r["table_rows"]
                }
                for i, r in enumerate(parsed_results)
            }
            st.download_button("📥 Download All Data", json.dumps(all_data, indent=2), file_name=f"{file.name}_all_forms.json")
