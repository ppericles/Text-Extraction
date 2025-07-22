# ============================================================
# FILE: app.py
# VERSION: 2.5.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Registry Form Parser with interactive canvas,
#              editable bounding boxes, OCR via Document AI,
#              fallback Vision API, and table extraction.
#              Includes per-form refresh to update crops.
# ============================================================

import streamlit as st
from PIL import Image
import os, json, tempfile

from streamlit_drawable_canvas import st_canvas
from utils_ocr import form_parser_ocr, match_fields_with_fallback, vision_api_ocr
from utils_image import optimize_image, resize_for_preview, trim_whitespace, split_zones_fixed
from utils_layout import draw_boxes, draw_column_breaks
from utils_parser import process_single_form

# === Config Setup ===
st.set_page_config(page_title="📄 Registry Parser", layout="wide")
st.title("📄 Registry Form Parser")

CONFIG_PATH = "config/processor_config.json"
os.makedirs("config", exist_ok=True)
default_config = {"project_id": "", "location": "", "processor_id": ""}
if os.path.exists(CONFIG_PATH):
    default_config = json.load(open(CONFIG_PATH))

st.sidebar.markdown("### 🔐 Credentials")
cred_file = st.sidebar.file_uploader("Upload Google JSON", type="json")
if cred_file:
    temp_path = tempfile.NamedTemporaryFile(delete=False)
    temp_path.write(cred_file.read()), temp_path.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path.name
    st.sidebar.success("✅ Credentials Loaded")

st.sidebar.markdown("### ⚙️ Document AI")
project_id = st.sidebar.text_input("Project ID", value=default_config["project_id"])
location = st.sidebar.text_input("Location", value=default_config["location"])
processor_id = st.sidebar.text_input("Processor ID", value=default_config["processor_id"])

if st.sidebar.button("💾 Save Config"):
    json.dump({
        "project_id": project_id.strip(),
        "location": location.strip(),
        "processor_id": processor_id.strip()
    }, open(CONFIG_PATH, "w"))
    st.sidebar.success("✅ Saved")

# === File Upload ===
uploaded_files = st.file_uploader("📤 Upload Registry Scans", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and all([project_id, location, processor_id]):
    config = {"project_id": project_id, "location": location, "processor_id": processor_id}

    for file in uploaded_files:
        st.header(f"📄 `{file.name}`")

        original = Image.open(file)
        processed = trim_whitespace(optimize_image(original.copy()))
        preview_img = resize_for_preview(original)

        st.markdown("### 🖼️ Preview Image")
        show_grayscale = st.checkbox("Show grayscale preview", value=False)
        display_img = resize_for_preview(processed) if show_grayscale else preview_img
        st.image(display_img, caption="📄 Preview", use_column_width=True)

        st.markdown("### ✏️ Bounding Box Editor")
        canvas_mode = st.radio("Canvas Mode", ["Draw", "Edit"], horizontal=True)
        drawing_mode = "rect" if canvas_mode == "Draw" else "transform"

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            background_image=preview_img,
            update_streamlit=True,
            height=preview_img.height,
            width=preview_img.width,
            drawing_mode=drawing_mode,
            key=f"canvas_{file.name}"
        )

        form_boxes = []
        if canvas_result.json_data:
            for obj in canvas_result.json_data["objects"]:
                x1 = obj["left"] / preview_img.width
                y1 = obj["top"] / preview_img.height
                x2 = (obj["left"] + obj["width"]) / preview_img.width
                y2 = (obj["top"] + obj["height"]) / preview_img.height
                form_boxes.append((x1, y1, x2, y2))

        st.markdown(f"### 📐 {len(form_boxes)} Form(s) Detected")

        for i, box in enumerate(form_boxes):
            st.subheader(f"🔍 Form {i+1} Results")

            if st.button(f"🔄 Refresh Form {i+1}", key=f"refresh_{i}"):
                st.experimental_rerun()

            auto = st.checkbox("Auto-detect table columns", value=True, key=f"auto_{i}")
            layout = {
                "master_ratio": 0.5,
                "group_a_box": [0.0, 0.0, 0.2, 1.0],
                "group_b_box": [0.2, 0.0, 1.0, 0.5],
                "detail_box": [0.0, 0.0, 1.0, 1.0],
                "auto_detect": auto
            }

            if not auto:
                st.markdown("📐 Define Table Columns")
                table_columns = []
                for c in range(6):
                    cx1 = st.slider(f"Column {c+1} - X1", 0.0, 1.0, c * 0.15, 0.01, key=f"cx1_{i}_{c}")
                    cx2 = st.slider(f"Column {c+1} - X2", 0.0, 1.0, (c + 1) * 0.15, 0.01, key=f"cx2_{i}_{c}")
                    table_columns.append((cx1, cx2))
                layout["table_columns"] = table_columns

            result = process_single_form(processed, box, i, config, layout)

            st.image(resize_for_preview(result["master"]), caption="🟦 Master Zone", use_column_width=True)

            overlay = result["detail"].copy()
            overlay = draw_column_breaks(overlay, result["column_breaks"])
            st.image(resize_for_preview(overlay), caption="📐 Table Column Breaks", use_column_width=True)

            st.markdown("### 🧾 Group A (ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ)")
            for label, data in result["group_a"].items():
                emoji = "🟢" if data["confidence"] >= 90 else "🟡" if data["confidence"] >= 70 else "🔴"
                st.text(f"{emoji} {label}: {data['value']} ({data['confidence']}%)")

            st.markdown("### 🧾 Group B")
            for label, data in result["group_b"].items():
                emoji = "🟢" if data["confidence"] >= 90 else "🟡" if data["confidence"] >= 70 else "🔴"
                st.text(f"{emoji} {label}: {data['value']} ({data['confidence']}%)")

            st.markdown("### 📊 Parsed Table Rows")
            if result["table_rows"]:
                st.dataframe(result["table_rows"], use_container_width=True)
            else:
                st.warning("⚠️ No table rows extracted.")
