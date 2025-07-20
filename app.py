import streamlit as st
from PIL import Image
from io import BytesIO
import os
import json
import tempfile

from utils_ocr import form_parser_ocr, match_fields_with_fallback
from utils_image import (
    resize_for_preview,
    trim_whitespace,
    split_zones_fixed,
    split_master_zone_vertically,
    draw_zones_overlays
)

# ==== Config Setup ====
CONFIG_PATH = "config/processor_config.json"
os.makedirs("config", exist_ok=True)

default_config = {"project_id": "", "location": "", "processor_id": ""}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        default_config = json.load(f)

# ==== Page Setup ====
st.set_page_config(page_title="📄 Registry Parser", layout="wide")
st.title("📄 Registry Form Parser")

# ==== Credential Upload ====
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

# ==== Expected Fields ====
expected_fields = {
    "group_a": ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"],
    "group_b": ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]
}

# ==== Main Logic ====
if uploaded_files and project_id and location and processor_id:
    for file in uploaded_files:
        base_name = file.name.replace(".", "_")
        st.header(f"📄 `{file.name}` — Parse & Extract")

        image = Image.open(file)
        clean = trim_whitespace(image)

        # Split zones
        split_ratio = st.slider("📐 Vertical split ratio for master zone", 0.2, 0.8, value=0.3, step=0.01, key=f"split_{base_name}")
        zones, bounds = split_zones_fixed(clean, master_ratio=0.5)
        master_zone = zones[0]

        preview = draw_zones_overlays(clean, bounds)
        st.image(resize_for_preview(preview), caption="📐 Zones Preview", use_column_width=True)

        # Split master zone into groups
        group_a, group_b = split_master_zone_vertically(master_zone, split_ratio)

        col1, col2 = st.columns(2)
        col1.image(resize_for_preview(group_a), caption="🟦 Group A: ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", use_column_width=True)
        col2.image(resize_for_preview(group_b), caption="🟩 Group B: Other Fields", use_column_width=True)

        # Load layout JSONs if available
        layout_dicts = {}
        for group_id in ["group_a", "group_b"]:
            layout_path = f"saved-layouts/{base_name}_{group_id}_layout.json"
            if os.path.exists(layout_path):
                with open(layout_path) as f:
                    layout_dicts[group_id] = json.load(f)
                st.sidebar.success(f"🧩 Loaded layout for {group_id}")
            else:
                layout_dicts[group_id] = {}

        # Run Form Parser OCR
        fields_a = form_parser_ocr(group_a, project_id, location, processor_id)
        fields_b = form_parser_ocr(group_b, project_id, location, processor_id)

        # Match with fallback
        matched_a = match_fields_with_fallback(expected_fields["group_a"], fields_a, group_a, layout_dicts.get("group_a", {}))
        matched_b = match_fields_with_fallback(expected_fields["group_b"], fields_b, group_b, layout_dicts.get("group_b", {}))

        # Display results
        def show_results(title, matched_fields):
            st.markdown(f"### 🧾 {title}")
            for label, data in matched_fields.items():
                value = data["value"]
                confidence = data["confidence"]
                if confidence >= 90:
                    emoji = "🟢"
                elif confidence >= 70:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                st.text(f"{emoji} {label}: {value}  ({confidence}%)")

        show_results("Group A (ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ)", matched_a)
        show_results("Group B (Remaining Fields)", matched_b)
