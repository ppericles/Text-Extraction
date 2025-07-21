# ==== FILE: app.py - Streamlit UI for Registry Form Parser ====
# Version: 1.5.0
# Created: 2025-07-21
# Author: Pericles & Copilot
# Description: Uses Document AI v1 API. Canvas removed. Layout editor uses sliders and bounding box previews.

import streamlit as st
from PIL import Image, ImageDraw
import os
import json
import tempfile

from utils_ocr import form_parser_ocr, match_fields_with_fallback
from utils_image import (
    optimize_image,
    resize_for_preview,
    trim_whitespace,
    split_zones_fixed,
    split_master_zone_vertically,
    draw_colored_zones
)

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

expected_fields = {
    "group_a": ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"],
    "group_b": ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]
}

if uploaded_files and project_id and location and processor_id:
    for file in uploaded_files:
        base_name = file.name.replace(".", "_")
        st.header(f"📄 `{file.name}` — Parse & Extract")

        image = Image.open(file)
        image = optimize_image(image)
        clean = trim_whitespace(image)

        # ==== Diagnostics ====
        st.write("🧪 Image diagnostics:")
        st.write(f"Mode: {clean.mode}")
        st.write(f"Size: {clean.size}")
        st.write(f"Type: {type(clean)}")

        if not isinstance(clean, Image.Image):
            st.error("❌ `clean` is not a PIL.Image — cannot proceed.")
            st.stop()
        if clean.mode != "RGB":
            clean = clean.convert("RGB")
        if clean.size[0] == 0 or clean.size[1] == 0:
            st.error("❌ Image size is invalid (0 width or height).")
            st.stop()

        # ==== Zone Splitting ====
        split_ratio = st.slider("📐 Vertical split ratio for master zone", 0.2, 0.8, value=0.3, step=0.01, key=f"split_{base_name}")
        zones, bounds = split_zones_fixed(clean, master_ratio=0.5)
        master_zone, detail_zone = zones
        master_bounds, detail_bounds = bounds

        group_a, group_b = split_master_zone_vertically(master_zone, split_ratio)

        col1, col2 = st.columns(2)
        col1.image(resize_for_preview(group_a), caption="🟦 Group A: ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", use_column_width=True)
        col2.image(resize_for_preview(group_b), caption="🟩 Group B: Other Fields", use_column_width=True)

        # ==== Layout Editor ====
        st.markdown("### ✏️ Define Field Zones with Sliders")
        edited_layout = {}
        field_types = ["Name", "Parent Name", "ID", "Date", "Location", "Custom"]

        for label in expected_fields["group_a"] + expected_fields["group_b"]:
            with st.expander(f"🗂️ Field: `{label}`", expanded=False):
                x1 = st.slider(f"{label} - X1", 0.0, 1.0, 0.1, 0.01, key=f"x1_{label}")
                y1 = st.slider(f"{label} - Y1", 0.0, 1.0, 0.1, 0.01, key=f"y1_{label}")
                x2 = st.slider(f"{label} - X2", 0.0, 1.0, 0.3, 0.01, key=f"x2_{label}")
                y2 = st.slider(f"{label} - Y2", 0.0, 1.0, 0.2, 0.01, key=f"y2_{label}")
                selected_type = st.selectbox("Field Type", field_types, key=f"type_{label}")
                edited_layout[label] = {
                    "box": [x1, y1, x2, y2],
                    "type": selected_type
                }

        # ==== Preview Bounding Boxes ====
        preview = clean.copy()
        draw = ImageDraw.Draw(preview)
        w, h = preview.size
        for label, meta in edited_layout.items():
            box = meta["box"]
            x1, y1, x2, y2 = box
            draw.rectangle((x1 * w, y1 * h, x2 * w, y2 * h), outline="red", width=2)
            draw.text((x1 * w, y1 * h), label, fill="red")

        st.image(resize_for_preview(preview), caption="🖼️ Field Layout Preview", use_column_width=True)

        if st.button("💾 Save Layout"):
            save_path = f"exports/layout_versions/{base_name}_layout.json"
            with open(save_path, "w") as f:
                json.dump(edited_layout, f, indent=2)
            st.success("✅ Layout saved.")

        # ==== OCR & Matching ====
        fields_a = form_parser_ocr(group_a, project_id, location, processor_id)
        fields_b = form_parser_ocr(group_b, project_id, location, processor_id)

        matched_a = match_fields_with_fallback(expected_fields["group_a"], fields_a, group_a, edited_layout)
        matched_b = match_fields_with_fallback(expected_fields["group_b"], fields_b, group_b, edited_layout)

        def show_results(title, matched_fields):
            st.markdown(f"### 🧾 {title}")
            for label, data in matched_fields.items():
                value = data["value"]
                confidence = data["confidence"]
                emoji = "🟢" if confidence >= 90 else "🟡" if confidence >= 70 else "🔴"
                st.text(f"{emoji} {label}: {value}  ({confidence}%)")

        show_results("Group A (ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ)", matched_a)
        show_results("Group B (Remaining Fields)", matched_b)
