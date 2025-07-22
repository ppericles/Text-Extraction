# ============================================================
# FILE: app.py
# VERSION: 3.7.9
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Registry Form Parser with interactive canvas,
#              bounding box editing, internal layout logic,
#              OCR via Vision API or Document AI, table parsing,
#              encrypted multi-profile config, adaptive trimming,
#              and batch export.
# ============================================================

import streamlit as st
from PIL import Image
import os, json, tempfile
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
from cryptography.fernet import Fernet

from utils_image import (
    resize_for_preview,
    trim_whitespace,
    adaptive_trim_whitespace,
    split_zones_fixed,
    draw_column_breaks,
    draw_row_breaks
)
from utils_layout import (
    extract_fields_from_layout,
    draw_layout_overlay
)
from utils_parser import process_single_form

# === Helper: Convert Boxes to Canvas Format ===
def convert_boxes_to_canvas_objects(boxes, scale=1.0):
    try:
        objects = []
        for box in boxes:
            x1, y1, x2, y2 = box
            left = x1 * scale
            top = y1 * scale
            width = (x2 - x1) * scale
            height = (y2 - y1) * scale
            obj = {
                "type": "rect",
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "fill": "rgba(255, 0, 0, 0.3)",
                "stroke": "red",
                "strokeWidth": 2
            }
            objects.append(obj)
        return {"objects": objects}
    except Exception as e:
        st.warning(f"⚠️ Failed to convert boxes: {e}")
        return {"objects": []}

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

# === Encrypted Config ===
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
        encrypted = open(ENC_PATH, "rb").read()
        decrypted = fernet.decrypt(encrypted).decode()
        saved_profiles = json.loads(decrypted)
    except Exception:
        saved_profiles = {}

default_profile = ""
if os.path.exists(LAST_PATH):
    default_profile = open(LAST_PATH).read().strip()

profile_names = list(saved_profiles.keys())
selected_profile = st.sidebar.selectbox("🔖 Select Profile", profile_names + ["New Profile"], index=profile_names.index(default_profile) if default_profile in profile_names else len(profile_names))

if selected_profile == "New Profile":
    st.sidebar.markdown("### ➕ Create New Profile")
    new_name = st.sidebar.text_input("Profile Name")
    project_id = st.sidebar.text_input("Project ID")
    location = st.sidebar.text_input("Location")
    processor_id = st.sidebar.text_input("Processor ID")

    if st.sidebar.button("💾 Save Profile", key="save_profile"):
        if new_name and project_id and location and processor_id:
            saved_profiles[new_name] = {
                "project_id": project_id.strip(),
                "location": location.strip(),
                "processor_id": processor_id.strip()
            }
            encrypted = fernet.encrypt(json.dumps(saved_profiles).encode())
            open(ENC_PATH, "wb").write(encrypted)
            open(LAST_PATH, "w").write(new_name)
            st.sidebar.success(f"✅ Profile `{new_name}` saved.")
            st.experimental_rerun()
        else:
            st.sidebar.error("⚠️ Fill in all fields.")

    st.sidebar.markdown("### 📋 Auto-Fill from JSON Paste or Upload")
    pasted_json = st.sidebar.text_area("Paste JSON", height=100, key="profile_clipboard")
    uploaded_profile = st.sidebar.file_uploader("Upload Profile JSON", type=["json"], key="profile_file")

    def load_profile(name, data):
        project_id = data.get("project_id", "")
        location = data.get("location", "")
        processor_id = data.get("processor_id", "")
        if name and project_id and location and processor_id:
            saved_profiles[name] = {
                "project_id": project_id.strip(),
                "location": location.strip(),
                "processor_id": processor_id.strip()
            }
            encrypted = fernet.encrypt(json.dumps(saved_profiles).encode())
            open(ENC_PATH, "wb").write(encrypted)
            open(LAST_PATH, "w").write(name)
            st.sidebar.success(f"✅ Profile `{name}` loaded.")
            st.experimental_rerun()
        else:
            st.sidebar.warning("⚠️ Missing fields or name.")

    if st.sidebar.button("📥 Load from Paste", key="load_clip"):
        try:
            data = json.loads(pasted_json)
            load_profile(new_name, data)
        except:
            st.sidebar.error("❌ Invalid JSON.")

    if uploaded_profile:
        try:
            data = json.load(uploaded_profile)
            load_profile(new_name, data)
        except:
            st.sidebar.error("❌ Failed to parse uploaded file.")
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

    if st.sidebar.button("🗑️ Delete Profile", key="delete_profile"):
        del saved_profiles[selected_profile]
        encrypted = fernet.encrypt(json.dumps(saved_profiles).encode())
        open(ENC_PATH, "wb").write(encrypted)
        open(LAST_PATH, "w").write("")
        st.sidebar.success(f"🗑️ Profile `{selected_profile}` deleted.")
        st.experimental_rerun()

# === Image Settings ===
st.sidebar.markdown("### 🖼️ Image Settings")
use_adaptive_trim = st.sidebar.checkbox("Use Adaptive Trimming", value=True)

# === File Upload ===
uploaded_files = st.file_uploader("📤 Upload Registry Scans", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if "saved_boxes" not in st.session_state:
    st.session_state.saved_boxes = {}
if "parsed_forms" not in st.session_state:
    st.session_state.parsed_forms = {}
if uploaded_files:
    for file in uploaded_files:
        st.header(f"📄 `{file.name}` — Select Forms")

        image_raw = Image.open(file).convert("RGB")
        processed = adaptive_trim_whitespace(image_raw.copy()) if use_adaptive_trim else trim_whitespace(image_raw.copy())
        preview_img = resize_for_preview(processed)

        st.image(preview_img, caption="Preview Image", use_column_width=True)

        # === Load existing boxes into canvas ===
        form_boxes = st.session_state.saved_boxes.get(file.name, [])
        try:
            scale = 1.0 / (processed.width / preview_img.width)
            canvas_json = convert_boxes_to_canvas_objects(form_boxes, scale=scale) if form_boxes else {"objects": []}
        except Exception as e:
            st.warning(f"⚠️ Canvas conversion error: {e}")
            canvas_json = {"objects": []}

        st.markdown("### ✏️ Draw or Edit Bounding Boxes")
        canvas_result = st_canvas(
            background_image=preview_img,
            initial_drawing=canvas_json,
            drawing_mode="rect",
            drawing_mode_selector=True,
            display_toolbar=True,
            editable=True,
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            height=preview_img.height,
            width=preview_img.width,
            update_streamlit=True,
            key=f"canvas_{file.name}"
        )

        # === Save updated boxes ===
        updated_boxes = []
        if canvas_result.json_data:
            scale_x = processed.width / preview_img.width
            scale_y = processed.height / preview_img.height
            for obj in canvas_result.json_data["objects"]:
                try:
                    x1 = int(obj["left"] * scale_x)
                    y1 = int(obj["top"] * scale_y)
                    x2 = int((obj["left"] + obj["width"]) * scale_x)
                    y2 = int((obj["top"] + obj["height"]) * scale_y)
                    updated_boxes.append((x1, y1, x2, y2))
                except Exception as e:
                    st.warning(f"⚠️ Error reading box: {e}")
            st.session_state.saved_boxes[file.name] = updated_boxes

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

            overlay = draw_layout_overlay(form_crop, layout)
            st.image(resize_for_preview(overlay), caption="🔍 Layout Overlay", use_column_width=True)

            column_overlay = draw_column_breaks(result["table_crop"], result["column_breaks"])
            row_overlay = draw_row_breaks(result["table_crop"], rows=10, header=True)
            st.image(resize_for_preview(column_overlay), caption="📊 Column Breaks", use_column_width=True)
            st.image(resize_for_preview(row_overlay), caption="📏 Row Breaks", use_column_width=True)

            st.markdown("### 🧾 Group A")
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
            batch_json = json.dumps(all_data, indent=2)
            st.download_button("📥 Download All Data", batch_json, file_name=f"{file.name}_all_forms.json")

# === Batch OCR with Progress Bar ===
if st.button("🚀 Run Batch OCR on All Files", key="run_batch_ocr"):
    total_forms = sum(len(st.session_state.saved_boxes.get(f.name, [])) for f in uploaded_files)
    progress = st.progress(0, text="Processing forms...")
    completed = 0
    st.session_state.parsed_forms = {}

    for file in uploaded_files:
        image_raw = Image.open(file).convert("RGB")
        processed = adaptive_trim_whitespace(image_raw.copy()) if use_adaptive_trim else trim_whitespace(image_raw.copy())
        form_boxes = st.session_state.saved_boxes.get(file.name, [])
        parsed_results = []

        for i, box in enumerate(form_boxes):
            x1, y1, x2, y2 = box
            form_crop = processed.crop((x1, y1, x2, y2))
            layout = {
                "master_ratio": 0.5,
                "group_a_box": [0.0, 0.0, 0.2, 1.0],
                "group_b_box": [0.2, 0.0, 1.0, 0.5],
                "detail_box": [0.0, 0.0, 1.0, 1.0],
                "auto_detect": True
            }
            config = docai_config if use_docai else {}
            result = process_single_form(form_crop, i, config, layout)
            parsed_results.append(result)

            completed += 1
            progress.progress(completed / total_forms, text=f"Processed {completed} of {total_forms} forms")

        st.session_state.parsed_forms[file.name] = parsed_results

    progress.empty()
    st.success("✅ Batch OCR completed.")
