# ==== FILE: app.py ====

import streamlit as st
from PIL import Image, ImageDraw
from io import BytesIO
import os
import json
import tempfile
import base64

from streamlit_drawable_canvas import st_canvas

from components.image_cropper import crop_and_confirm_forms
from utils_image import (
    trim_whitespace,
    split_zones_fixed,
    draw_zones_overlays,
    draw_layout_overlay,
    draw_invalid_boxes_overlay,
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
from utils_ocr import (
    parse_zone_text,
    extract_fields_from_layout
)
from utils_text import preview_metadata_row


def get_fallback_image(width=400, height=200, text="Zone image unavailable"):
    img = Image.new("RGB", (width, height), color="lightgray")
    draw = ImageDraw.Draw(img)
    draw.text((10, height // 2 - 10), text, fill="black")
    return img


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


# ==== File Upload ====
uploaded_files = st.file_uploader(
    "📤 Upload Registry Scans",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)


# ==== Main Logic ====
if uploaded_files:
    for file in uploaded_files:
        base_name = file.name.replace(".", "_")
        st.header(f"📄 `{file.name}` — Crop & Parse")

        image = Image.open(file)
        confirmed_forms = crop_and_confirm_forms(image, max_crops=5)

        for idx, img in enumerate(confirmed_forms, start=1):
            form_id = f"{base_name}_form_{idx}"
            st.subheader(f"🧾 Form `{form_id}`")

            clean = trim_whitespace(img)

            slider_key = f"split_slider_{form_id}"
            if slider_key not in st.session_state:
                st.session_state[slider_key] = 0.5
            master_ratio = st.slider(
                "Adjust vertical split",
                0.0, 1.0,
                value=st.session_state[slider_key],
                step=0.01,
                key=slider_key
            )

            zones, bounds = split_zones_fixed(clean, master_ratio=master_ratio)
            preview = draw_zones_overlays(clean, bounds)
            st.image(resize_for_preview(preview), caption=f"📐 Zones for `{form_id}`", use_column_width=True)

            layout_dicts = {}
            save_dir = "saved-layouts"
            os.makedirs(save_dir, exist_ok=True)

            for zid in ["1", "2"]:
                st.markdown(f"### 🧱 Zone {zid} Layout Editor")
                zone_img = zones[int(zid) - 1]

                if not isinstance(zone_img, Image.Image) or zone_img.size == (0, 0):
                    zone_img = get_fallback_image(text=f"Zone {zid} unavailable")

                zone_img = zone_img.convert("RGB")

                try:
                    canvas_result = st_canvas(
                        fill_color="rgba(0, 255, 0, 0.3)",
                        stroke_width=3,
                        background_image=zone_img,
                        update_streamlit=True,
                        height=zone_img.size[1],
                        width=zone_img.size[0],
                        drawing_mode="rect",
                        key=f"canvas_{form_id}_{zid}"
                    )

                    def convert_to_layout_dict(objects, image_size):
                        layout = {}
                        w, h = image_size
                        for i, obj in enumerate(objects):
                            if obj["type"] == "rect":
                                left = obj["left"] / w
                                top = obj["top"] / h
                                width = obj["width"] / w
                                height = obj["height"] / h
                                layout[f"field_{i}"] = [left, top, left + width, top + height]
                        return layout

                    if canvas_result.json_data and "objects" in canvas_result.json_data:
                        layout_dict = convert_to_layout_dict(canvas_result.json_data["objects"], zone_img.size)
                        layout_dicts[zid] = layout_dict

                        overlay = draw_layout_overlay(zone_img, layout_dict)
                        st.image(resize_for_preview(overlay), caption=f"🔍 Zone {zid} Overlay", use_column_width=True)

                        json_str = json.dumps(layout_dict, indent=2)
                        json_path = f"{save_dir}/{form_id}_zone_{zid}_layout.json"
                        with open(json_path, "w") as f:
                            f.write(json_str)
                        st.download_button(f"💾 Download Layout JSON", json_str, file_name=os.path.basename(json_path))
                        st.sidebar.success(f"📝 Layout saved: `{json_path}`")

                        debug_overlay = draw_invalid_boxes_overlay(zone_img, layout_dict)
                        st.image(resize_for_preview(debug_overlay), caption=f"🚨 Invalid Fields in Zone {zid}", use_column_width=True)

                except Exception as e:
                    st.warning(f"⚠️ Canvas failed with error: {e}. Switching to slider editor.")

                    field_count = st.number_input(f"Number of fields in Zone {zid}", min_value=1, max_value=10, value=3, key=f"field_count_{form_id}_{zid}")
                    layout_dict = {}

                    for i in range(field_count):
                        st.markdown(f"🧩 Field {i + 1}")
                        x1 = st.slider(f"x1 (left)", 0.0, 1.0, 0.05, 0.01, key=f"x1_{form_id}_{zid}_{i}")
                        y1 = st.slider(f"y1 (top)", 0.0, 1.0, 0.05, 0.01, key=f"y1_{form_id}_{zid}_{i}")
                        x2 = st.slider(f"x2 (right)", x1 + 0.01, 1.0, x1 + 0.3, 0.01, key=f"x2_{form_id}_{zid}_{i}")
                        y2 = st.slider(f"y2 (bottom)", y1 + 0.01, 1.0, y1 + 0.1, 0.01, key=f"y2_{form_id}_{zid}_{i}")
                        layout_dict[f"field_{i}"] = [x1, y1, x2, y2]

                    layout_dicts[zid] = layout_dict

                    overlay = draw_layout_overlay(zone_img, layout_dict)
                    st.image(resize_for_preview(overlay), caption=f"🔍 Manual Layout Preview", use_column_width=True)

                    json_str = json.dumps(layout_dict, indent=2)
                    st.download_button(f"💾 Download Manual Layout JSON", json_str, file_name=f"{form_id}_zone_{zid}_layout_manual.json", mime="application/json")

            ocr_traces = {}
            trace = []

            for zid in ["1", "2", "3"]:
                zone_img = zones[int(zid) - 1]
                if zone_img is not None:
                    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                        zone_ocr = parse_zone_text(zone_img, engine="vision")
                    else:
                        zone_ocr = "⚠️ OCR skipped."
                else:
                    zone_ocr = f"⚠️ Zone {zid} missing."
                trace.append(zone_ocr)

            ocr_traces[form_id] = trace

            extracted_fields = {}
            for zid in ["1", "2"]:
                zone_img = zones[int(zid) - 1]
                layout = layout_dicts.get(zid, {})
                fields = extract_fields_from_layout(zone_img, layout, engine="vision")
                extracted_fields.update(fields)

            st.markdown("### 🧾 Extracted Fields")
            for label, value in extracted_fields.items():
                st.text(f"{label}: {value}")

            mock_rows = generate_mock_metadata_batch(layout_dicts, {}, count=1, placeholder="XXXX")
            preview_metadata_row(mock_rows[0])

            export_mock_dataset_with_layout_overlay(
                mock_rows,
                zones,
                layout_dicts,
                ocr_traces,
                output_dir="training-set"
            )

            st.success(f"📁 Form `{form_id}` exported to `training-set/`")
