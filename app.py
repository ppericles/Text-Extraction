# ==== FILE: app.py - Streamlit UI for Registry Form Parser ====
# Version: 1.9.0
# Created: 2025-07-21
# Author: Pericles & Copilot
# Description: Parses 3 forms per image using bounding boxes. Uses Document AI v1 API. Table rows visualized with column overlays and auto-detection.

import streamlit as st
from PIL import Image, ImageDraw
import os
import json
import tempfile

from utils_ocr import form_parser_ocr, match_fields_with_fallback, vision_api_ocr
from utils_image import (
    optimize_image,
    resize_for_preview,
    trim_whitespace,
    split_zones_fixed,
    split_master_zone_vertically,
    draw_colored_zones
)

from reusable_form import process_single_form  # ← Your reusable function

# ==== Setup ====
CONFIG_PATH = "config/processor_config.json"
os.makedirs("config", exist_ok=True)
os.makedirs("exports/layout_versions", exist_ok=True)

default_config = {"project_id": "", "location": "", "processor_id": ""}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        default_config = json.load(f)

st.set_page_config(page_title="📄 Registry Parser", layout="wide")
st.title("📄 Registry Form Parser")

# ==== Credentials ====
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

# ==== Processor Config ====
st.sidebar.markdown("### ⚙️ Document AI Config")
project_id = st.sidebar.text_input("Project ID", value=default_config.get("project_id", ""))
location = st.sidebar.text_input("Location", value=default_config.get("location", ""))
processor_id = st.sidebar.text_input("Processor ID", value=default_config.get("processor_id", ""))

if st.sidebar.button("💾 Save Config"):
    new_config = {
        "project_id": project_id.strip(),
        "location": location.strip(),
        "processor_id": processor_id.strip()
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(new_config, f, indent=2)
    st.sidebar.success("✅ Configuration saved.")

# ==== File Upload ====
uploaded_files = st.file_uploader("📤 Upload Registry Scans", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and project_id and location and processor_id:
    config = {
        "project_id": project_id,
        "location": location,
        "processor_id": processor_id
    }

    for file in uploaded_files:
        base_name = file.name.replace(".", "_")
        st.header(f"📄 `{file.name}` — Parse & Extract")

        image = Image.open(file)
        image = optimize_image(image)
        clean = trim_whitespace(image)

        st.image(resize_for_preview(clean), caption="🖼️ Full Image", use_column_width=True)

        # ==== Define Form Bounding Boxes ====
        st.markdown("## 🧩 Define Form Bounding Boxes")
        form_boxes = []
        for i in range(3):
            with st.expander(f"📄 Form {i+1} Bounding Box", expanded=False):
                fx1 = st.slider(f"Form {i+1} - X1", 0.0, 1.0, 0.0, 0.01, key=f"fx1_{i}")
                fy1 = st.slider(f"Form {i+1} - Y1", 0.0, 1.0, i * 0.33, 0.01, key=f"fy1_{i}")
                fx2 = st.slider(f"Form {i+1} - X2", 0.0, 1.0, 1.0, 0.01, key=f"fx2_{i}")
                fy2 = st.slider(f"Form {i+1} - Y2", 0.0, 1.0, (i + 1) * 0.33, 0.01, key=f"fy2_{i}")
                form_boxes.append([fx1, fy1, fx2, fy2])

        # ==== Layout Dict Base ====
        base_layout = {
            "master_ratio": 0.5,
            "group_a_box": [0.0, 0.0, 0.2, 1.0],
            "group_b_box": [0.2, 0.0, 1.0, 0.5],
            "detail_box": [0.0, 0.0, 1.0, 1.0]
        }

        # ==== Process Each Form ====
        for i, box in enumerate(form_boxes):
            st.markdown(f"### 🧩 Form {i+1} Table Column Boundaries")
            auto_detect = st.checkbox(f"Auto-detect columns for Form {i+1}", value=True, key=f"auto_{i}")

            layout_dict = base_layout.copy()
            layout_dict["auto_detect_columns"] = auto_detect

            if not auto_detect:
                table_columns = []
                for col_idx in range(6):
                    x1 = st.slider(f"Column {col_idx+1} - X1", 0.0, 1.0, col_idx * 0.15, 0.01, key=f"col_x1_{i}_{col_idx}")
                    x2 = st.slider(f"Column {col_idx+1} - X2", 0.0, 1.0, (col_idx + 1) * 0.15, 0.01, key=f"col_x2_{i}_{col_idx}")
                    table_columns.append((x1, x2))
                layout_dict["table_columns"] = table_columns

            result = process_single_form(clean, box, i, config, layout_dict)

            with st.expander(f"📋 Form {i+1} Results", expanded=True):
                st.image(resize_for_preview(result["master_zone"]), caption="🟦 Master Zone", use_column_width=True)

                # Column overlay preview
                preview = result["detail_zone"].copy()
                draw = ImageDraw.Draw(preview)
                w, h = preview.size
                for x1, x2 in result["column_breaks"]:
                    x = int(x1 * w)
                    draw.line([(x, 0), (x, h)], fill="red", width=2)
                st.image(resize_for_preview(preview), caption="📐 Column Break Preview", use_column_width=True)

                st.markdown("### 🧾 Group A (ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ)")
                for label, data in result["group_a"].items():
                    emoji = "🟢" if data["confidence"] >= 90 else "🟡" if data["confidence"] >= 70 else "🔴"
                    st.text(f"{emoji} {label}: {data['value']} ({data['confidence']}%)")

                st.markdown("### 🧾 Group B (4 Fields)")
                for label, data in result["group_b"].items():
                    emoji = "🟢" if data["confidence"] >= 90 else "🟡" if data["confidence"] >= 70 else "🔴"
                    st.text(f"{emoji} {label}: {data['value']} ({data['confidence']}%)")

                st.markdown("### 📊 Detail Zone Table")
                if result["table_rows"]:
                    st.dataframe(result["table_rows"], use_container_width=True)
                else:
                    st.warning("⚠️ No table rows extracted from detail zone.")
