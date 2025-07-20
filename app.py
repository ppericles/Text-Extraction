# ==== app.py ====

import streamlit as st
from PIL import Image
import os
import tempfile

from components.image_cropper import crop_and_confirm_forms
from utils_image import (
    trim_whitespace,
    split_zones_fixed,
    draw_zones_overlays,
    draw_layout_overlay,
    resize_for_preview
)
from utils_layout import (
    LayoutManager,
    ensure_zone_layout,
    load_default_layout
)
from utils_mock import (
    generate_mock_metadata_batch,
    export_mock_dataset_with_layout_overlay
)
from utils_ocr import parse_zone_text
from utils_text import preview_metadata_row

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
    st.sidebar.success("✅ Credentials loaded successfully.")
else:
    st.sidebar.warning("⚠️ Upload service account JSON to enable OCR.")

# ==== File Upload ====
uploaded_files = st.file_uploader(
    "📤 Upload Registry Scans with Multiple Forms Per Image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

default_templates = {
    "1": "templates/layout_zone_1.json",
    "2": "templates/layout_zone_2.json"
}

# ==== Main Processing ====
if uploaded_files:
    for file in uploaded_files:
        base_name = file.name.replace(".", "_")
        st.header(f"📄 `{file.name}` — Multi-Form Cropping")

        image = Image.open(file)
        confirmed_forms = crop_and_confirm_forms(image, max_crops=5)

        for idx, img in enumerate(confirmed_forms, start=1):
            form_id = f"{base_name}_form_{idx}"
            st.subheader(f"🧾 Processing `{form_id}`")

            # Preprocessing
            clean = trim_whitespace(img)
            zones, bounds = split_zones_fixed(clean)
            preview = draw_zones_overlays(clean, bounds)
            st.image(resize_for_preview(preview), caption=f"📐 Zones for `{form_id}`", use_column_width=True)

            # Layout setup
            layout_managers = {}
            box_layouts = {}
            expected_labels = {
                "1": ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"],
                "2": ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"],
                "3": []  # Table zone
            }

            for zid in ["1", "2"]:
                st.markdown(f"### 🧱 Zone {zid} Layout — `{form_id}`")
                img_zone = zones[int(zid) - 1]
                manager = LayoutManager(img_zone.size)
                layout_managers[zid] = manager
                default_template = load_default_layout(zid, default_templates)
                box_layouts[zid] = manager.save_layout(default_template)

                overlay = draw_layout_overlay(img_zone, box_layouts[zid])
                st.image(resize_for_preview(overlay), caption=f"🔍 Zone {zid} Overlay", use_column_width=True)
                ensure_zone_layout(zid, expected_labels[zid], layout_managers, box_layouts, st)

            # OCR + Metadata
            ocr_traces = {}
            trace = []

            for zid in ["1", "2", "3"]:
                zone_img = zones[int(zid) - 1]
                if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                    zone_ocr = parse_zone_text(zone_img, engine="vision")
                else:
                    zone_ocr = "⚠️ OCR skipped — no credentials loaded."
                trace.append(zone_ocr)

            ocr_traces[form_id] = trace

            mock_rows = generate_mock_metadata_batch(box_layouts, expected_labels, count=1, placeholder="XXXX")
            preview_metadata_row(mock_rows[0])

            export_mock_dataset_with_layout_overlay(
                mock_rows,
                zones,
                box_layouts,
                ocr_traces,
                output_dir="training-set"
            )

            st.success(f"📁 Exported `{form_id}` to `training-set/`")
