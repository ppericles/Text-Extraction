# ==== FILE: app.py - Streamlit UI for Registry Form Parser ====
# Version: 1.0.0
# Created: 2025-07-21
# Author: Pericles & Copilot
# Description: Handles UI, file uploads, OCR pipeline, layout editing, and result display.

import streamlit as st
from PIL import Image
import io
import os
import json
import tempfile

from streamlit_drawable_canvas import st_canvas
from utils_ocr import form_parser_ocr, match_fields_with_fallback
from utils_image import (
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
        clean = trim_whitespace(image)

        # ==== Canvas Safety Guard ====
        if not isinstance(clean, Image.Image):
            st.error("🧯 Invalid image format — expected a PIL.Image.")
            st.stop()
        if clean.mode != "RGB":
            clean = clean.convert("RGB")

        # ==== Zone Splitting ====
        split_ratio = st.slider("📐 Vertical split ratio for master zone", 0.2, 0.8, value=0.3, step=0.01, key=f"split_{base_name}")
        zones, bounds = split_zones_fixed(clean, master_ratio=0.5)
        master_zone, detail_zone = zones
        master_bounds, detail_bounds = bounds

        w_m, h_m = master_zone.size
        split_x = int(w_m * split_ratio)
        group_bounds = {
            "group_a": (master_bounds[0], master_bounds[1], master_bounds[0] + split_x, master_bounds[3]),
            "group_b": (master_bounds[0] + split_x, master_bounds[1], master_bounds[2], master_bounds[3]),
        }

        overlay = draw_colored_zones(clean, master_bounds, detail_bounds, group_bounds)
        st.image(resize_for_preview(overlay), caption="📐 Zone Debug Overlay", use_column_width=True)

        group_a, group_b = split_master_zone_vertically(master_zone, split_ratio)

        col1, col2 = st.columns(2)
        col1.image(resize_for_preview(group_a), caption="🟦 Group A: ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", use_column_width=True)
        col2.image(resize_for_preview(group_b), caption="🟩 Group B: Other Fields", use_column_width=True)

        # ==== Interactive Canvas ====
        st.markdown("### ✏️ Draw Field Zones")
        try:
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 255, 0.2)",
                stroke_width=3,
                background_image=clean,
                update_streamlit=True,
                height=clean.height,
                width=clean.width,
                drawing_mode="rect",
                key=f"canvas_{base_name}"
            )
        except Exception as e:
            st.error("🧯 Canvas failed to load. Please check the image format or reload the app.")
            st.stop()

        # ==== Convert Canvas to Layout ====
        layout_dict = {}
        if canvas_result.json_data:
            for obj in canvas_result.json_data["objects"]:
                label = obj.get("name", f"field_{len(layout_dict)}")
                x1 = obj["left"] / clean.width
                y1 = obj["top"] / clean.height
                x2 = (obj["left"] + obj["width"]) / clean.width
                y2 = (obj["top"] + obj["height"]) / clean.height
                layout_dict[label] = [x1, y1, x2, y2]

        # ==== Sidebar Field Editor ====
        st.sidebar.markdown("### ✏️ Edit Field Zones")
        edited_layout = {}
        field_types = ["Name", "Parent Name", "ID", "Date", "Location", "Custom"]

        for field_label, box_coords in layout_dict.items():
            with st.sidebar.expander(f"🗂️ Zone: `{field_label}`", expanded=False):
                new_label = st.text_input("Label", value=field_label, key=f"label_{field_label}")
                selected_type = st.selectbox("Field Type", field_types, key=f"type_{field_label}")
                edited_layout[new_label] = {
                    "box": box_coords,
                    "type": selected_type
                }

        if st.sidebar.button("💾 Save Edited Layout"):
            save_path = f"exports/layout_versions/{base_name}_layout.json"
            with open(save_path, "w") as f:
                json.dump(edited_layout, f, indent=2)
            st.sidebar.success("✅ Edited layout saved.")

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
