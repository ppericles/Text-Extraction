# ============================================================
# FILE: app.py
# VERSION: 3.7.12
# DESCRIPTION: Streamlit-based Registry Form Parser with
#              interactive canvas drawing, multi-profile
#              config encryption, Google OCR (Vision API or
#              Document AI), layout overlays, batch export,
#              and resilience against rendering errors.
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
    draw_column_breaks,
    draw_row_breaks
)
from utils_layout import draw_layout_overlay
from utils_parser import process_single_form

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
        st.warning(f"⚠️ Box conversion error: {e}")
        return {"objects": []}

st.set_page_config(page_title="📄 Registry Parser", layout="wide")
st.title("📄 Registry Form Parser")

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

st.sidebar.markdown("### 🧠 OCR Engine")
ocr_engine = st.sidebar.radio("Choose OCR Engine", ["Vision API", "Document AI"])
use_docai = ocr_engine == "Document AI"

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
        form_boxes = st.session_state.saved_boxes[file.name]
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
                table_columns = []
                for c in range(6):
                    cx1 = st.slider(f"Column {c+1} - X1", 0.0, 1.0, c * 0.15, 0.01, key=f"cx1_{i}_{c}")
                    cx2 = st.slider(f"Column {c+1} - X2", 0.0, 1.0, (c + 1) * 0.15, 0.01, key=f"cx2_{i}_{c}")
                    table_columns.append((cx1, cx2))
                layout["table_columns"] = table_columns

            config = docai_config if use_docai else {}
            result = process_single_form(form_crop, i, config, layout)
            parsed_results.append(result)

            st.image(resize_for_preview(draw_layout_overlay(form_crop, layout)), caption="🔍 Layout Overlay", use_column_width=True)
            st.image(resize_for_preview(draw_column_breaks(result["table_crop"], result["column_breaks"])), caption="📊 Column Breaks", use_column_width=True)
            st.image(resize_for_preview(draw_row_breaks(result["table_crop"], rows=10, header=True)), caption="📏 Row Breaks", use_column_width=True)

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

            st.download_button("📥 Download Layout JSON", json.dumps(layout, indent=2), file_name=f"form_{i+1}_layout.json")
            buffer = BytesIO()
            form_crop.save(buffer, format="PNG")
            st.download_button("🖼️ Download Cropped Form", buffer.getvalue(), file_name=f"form_{i+1}.png")
            st.download_button("📤 Download Parsed Data", json.dumps({
                "group_a": result["group_a"],
                "group_b": result["group_b"],
                "table_rows": result["table_rows"]
            }, indent=2), file_name=f"form_{i+1}_data.json")

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
            st.download_button("📥 Download All Data", json.dumps(all_data, indent=2), file_name=f"{file.name}_all_forms.json")
# === Batch OCR with Progress Bar ===
if st.button("🚀 Run Batch OCR on All Files", key="run_batch_ocr"):
    total_forms = sum(len(st.session_state.saved_boxes.get(f.name, [])) for f in uploaded_files)
    progress = st.progress(0, text="Processing forms...")
    completed = 0
    st.session_state.parsed_forms = {}

    for file in uploaded_files:
        try:
            image_raw = Image.open(file).convert("RGB")
            processed = adaptive_trim_whitespace(image_raw.copy()) if use_adaptive_trim else trim_whitespace(image_raw.copy())
        except Exception as e:
            st.warning(f"⚠️ Failed to process `{file.name}`: {e}")
            continue

        form_boxes = st.session_state.saved_boxes.get(file.name, [])
        parsed_results = []

        for i, box in enumerate(form_boxes):
            try:
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
            except Exception as e:
                st.warning(f"⚠️ Error parsing form {i+1} in `{file.name}`: {e}")

            completed += 1
            progress.progress(completed / total_forms, text=f"Processed {completed} of {total_forms} forms")

        st.session_state.parsed_forms[file.name] = parsed_results

    progress.empty()
    st.success("✅ Batch OCR completed.")
