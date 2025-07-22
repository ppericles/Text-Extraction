# ============================================================
# FILE: app.py
# VERSION: 3.5.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Registry Form Parser with interactive canvas,
#              bounding box selection, internal layout logic,
#              OCR via Vision API or Document AI, table parsing,
#              session state persistence, and batch export.
# ============================================================

import streamlit as st
from PIL import Image
import os, json, tempfile
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

from utils_image import (
    resize_for_preview,
    draw_layout_overlay,
    split_zones_fixed,
    trim_whitespace
)
from utils_layout import extract_fields_from_layout
from utils_parser import process_single_form
from utils_ocr import form_parser_ocr, vision_api_ocr

# === Page Setup ===
st.set_page_config(page_title="📄 Registry Parser", layout="wide")
st.title("📄 Registry Form Parser")

# === Credential Upload ===
st.sidebar.markdown("### 🔐 Load Google Credentials")
cred_file = st.sidebar.file_uploader("Upload JSON credentials", type=["json"])
if cred_file:
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_path.write(cred_file.read())
    temp_path.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path.name
    st.sidebar.success("✅ Credentials loaded.")
else:
    st.sidebar.warning("⚠️ OCR disabled — upload a service account JSON.")

# === OCR Engine Selection ===
st.sidebar.markdown("### 🧠 OCR Engine")
ocr_engine = st.sidebar.radio("Choose OCR Engine", ["Vision API", "Document AI"])
use_docai = ocr_engine == "Document AI"

# === Document AI Config ===
project_id = st.sidebar.text_input("Project ID")
location = st.sidebar.text_input("Location")
processor_id = st.sidebar.text_input("Processor ID")

docai_config = {
    "project_id": project_id.strip(),
    "location": location.strip(),
    "processor_id": processor_id.strip()
}

# === File Upload ===
uploaded_files = st.file_uploader("📤 Upload Registry Scans", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# === Session State ===
if "saved_boxes" not in st.session_state:
    st.session_state.saved_boxes = {}
if "parsed_forms" not in st.session_state:
    st.session_state.parsed_forms = {}

# === Main Logic ===
if uploaded_files:
    for file in uploaded_files:
        st.header(f"📄 `{file.name}` — Select Forms")

        image = Image.open(file).convert("RGB")
        processed = trim_whitespace(image.copy())
        preview_img = resize_for_preview(image)

        st.markdown("### ✏️ Draw Bounding Boxes")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            background_image=preview_img,
            update_streamlit=True,
            height=preview_img.height,
            width=preview_img.width,
            drawing_mode="rect",
            key=f"canvas_{file.name}"
        )

        form_boxes = []
        if canvas_result.json_data:
            scale_x = processed.width / preview_img.width
            scale_y = processed.height / preview_img.height

            for obj in canvas_result.json_data["objects"]:
                x1 = int(obj["left"] * scale_x)
                y1 = int(obj["top"] * scale_y)
                x2 = int((obj["left"] + obj["width"]) * scale_x)
                y2 = int((obj["top"] + obj["height"]) * scale_y)
                form_boxes.append((x1, y1, x2, y2))

            st.session_state.saved_boxes[file.name] = form_boxes

        if file.name in st.session_state.saved_boxes:
            form_boxes = st.session_state.saved_boxes[file.name]

        st.markdown(f"### 📐 {len(form_boxes)} Form(s) Selected")

        parsed_results = []

        for i, box in enumerate(form_boxes):
            x1, y1, x2, y2 = box
            form_crop = processed.crop((x1, y1, x2, y2))
            st.subheader(f"🧾 Form {i+1}")
            st.image(resize_for_preview(form_crop), caption="📄 Cropped Form", use_column_width=True)

            st.markdown("### 🧩 Internal Layout Settings")
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

            config = docai_config if use_docai else {}
            result = process_single_form(form_crop, i, config, layout)
            parsed_results.append(result)

            st.image(resize_for_preview(result["master"]), caption="🟦 Master Zone", use_column_width=True)
            st.image(resize_for_preview(result["detail"]), caption="📘 Detail Zone", use_column_width=True)

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

            st.markdown("### 💾 Export Layout & Data")
            layout_json = json.dumps(layout, indent=2)
            st.download_button("📥 Download Layout JSON", layout_json, file_name=f"form_{i+1}_layout.json")

            buffer = BytesIO()
            form_crop.save(buffer, format="PNG")
            st.download_button("🖼️ Download Cropped Form", buffer.getvalue(), file_name=f"form_{i+1}.png")

            result_json = json.dumps({
                "group_a": result["group_a"],
                "group_b": result["group_b"],
                "table_rows": result["table_rows"]
            }, indent=2)
            st.download_button("📤 Download Parsed Data", result_json, file_name=f"form_{i+1}_data.json")

        st.session_state.parsed_forms[file.name] = parsed_results

        # === Batch Export ===
        st.markdown("## 📦 Export All Forms")
        if st.button("📤 Export All Parsed Data"):
            all_data = {
                f"form_{i+1}": {
                    "group_a": r["group_a"],
                    "group_b": r["group_b"],
                    "table_rows": r["table_rows"]
                }
                for i, r in enumerate(parsed_results)
            }
            batch_json = json.dumps(all_data, indent=2)
            st.download_button("📥 Download All Data", batch_json, file_name=f"{file.name}_all_forms.json")
