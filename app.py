# =============================================================================
# FILE: app.py
# VERSION: 4.0.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Streamlit Registry Parser with canvas editing,
#              Document AI profile management, layout overlays,
#              and manual OCR triggering.
# =============================================================================

import streamlit as st
from PIL import Image
import os, json, tempfile
from streamlit_drawable_canvas import st_canvas
from cryptography.fernet import Fernet

from utils_image import (
    resize_for_preview,
    trim_whitespace,
    draw_column_breaks,
    draw_row_breaks
)
from utils_layout import (
    draw_layout_overlay,
    validate_layout_for_preview,
    draw_layout_overlay_preview
)
from utils_parser import process_single_form
from form_cropper import crop_and_confirm_forms

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

# === Document AI Profile Management ===
CONFIG_DIR = "config"
ENC_PATH = os.path.join(CONFIG_DIR, "processor_config.enc")
LAST_PATH = os.path.join(CONFIG_DIR, "last_profile.txt")
KEY_PATH = os.path.join(CONFIG_DIR, "key.txt")
os.makedirs(CONFIG_DIR, exist_ok=True)

if not os.path.exists(KEY_PATH):
    key = Fernet.generate_key()
    open(KEY_PATH, "wb").write(key)
else:
    key = open(KEY_PATH, "rb").read()
fernet = Fernet(key)

saved_profiles = {}
if os.path.exists(ENC_PATH):
    try:
        decrypted = fernet.decrypt(open(ENC_PATH, "rb").read()).decode()
        saved_profiles = json.loads(decrypted)
    except:
        saved_profiles = {}

default_profile = ""
if os.path.exists(LAST_PATH):
    default_profile = open(LAST_PATH).read().strip()

profile_names = list(saved_profiles.keys())
selected_profile = st.sidebar.selectbox("🔖 Select Profile", profile_names + ["New Profile"],
    index=profile_names.index(default_profile) if default_profile in profile_names else len(profile_names)
)

if selected_profile == "New Profile":
    st.sidebar.markdown("### ➕ Create New Profile")
    new_name = st.sidebar.text_input("Profile Name")
    project_id = st.sidebar.text_input("Project ID")
    location = st.sidebar.text_input("Location")
    processor_id = st.sidebar.text_input("Processor ID")

    def save_profile(name, proj, loc, proc):
        saved_profiles[name] = {
            "project_id": proj.strip(),
            "location": loc.strip(),
            "processor_id": proc.strip()
        }
        encrypted = fernet.encrypt(json.dumps(saved_profiles).encode())
        open(ENC_PATH, "wb").write(encrypted)
        open(LAST_PATH, "w").write(name)
        st.sidebar.success(f"✅ Profile `{name}` saved.")
        st.experimental_rerun()

    if st.sidebar.button("💾 Save Profile"):
        if new_name and project_id and location and processor_id:
            save_profile(new_name, project_id, location, processor_id)
        else:
            st.sidebar.error("⚠️ Please fill in all fields.")

    st.sidebar.markdown("### 📋 Paste Profile JSON")
    pasted_json = st.sidebar.text_area("Paste JSON", height=100)
    if st.sidebar.button("📥 Load from Paste"):
        try:
            data = json.loads(pasted_json)
            save_profile(new_name, data.get("project_id", ""), data.get("location", ""), data.get("processor_id", ""))
        except:
            st.sidebar.error("❌ Invalid JSON format.")

    uploaded_profile = st.sidebar.file_uploader("Upload Profile JSON", type=["json"])
    if uploaded_profile:
        try:
            data = json.load(uploaded_profile)
            save_profile(new_name, data.get("project_id", ""), data.get("location", ""), data.get("processor_id", ""))
        except:
            st.sidebar.error("❌ Failed to parse uploaded profile.")
else:
    profile = saved_profiles.get(selected_profile) or {}
    if all(k in profile for k in ["project_id", "location", "processor_id"]):
        st.sidebar.markdown(f"### 📁 Profile: `{selected_profile}`")
        st.sidebar.text(f"Project ID: {profile['project_id']}")
        st.sidebar.text(f"Location: {profile['location']}")
        st.sidebar.text(f"Processor ID: {profile['processor_id']}")
        docai_config = profile
        open(LAST_PATH, "w").write(selected_profile)
    else:
        st.sidebar.warning("⚠️ Selected profile is incomplete.")
        docai_config = {}

    if st.sidebar.button("🗑️ Delete Profile"):
        del saved_profiles[selected_profile]
        encrypted = fernet.encrypt(json.dumps(saved_profiles).encode())
        open(ENC_PATH, "wb").write(encrypted)
        open(LAST_PATH, "w").write("")
        st.sidebar.success(f"🗑️ Profile `{selected_profile}` deleted.")
        st.experimental_rerun()

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
        base_name = file.name.replace(".", "_")
        st.header(f"📄 `{file.name}` — Crop & Parse")

        image = Image.open(file).convert("RGB")
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
                zone_img = zones[int(zid) - 1] if int(zid) - 1 < len(zones) else None
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
