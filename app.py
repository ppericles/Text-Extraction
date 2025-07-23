# =============================================================================
# FILE: app.py
# VERSION: 5.1.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Registry Form Parser with Document AI support,
#              interactive cropping, layout refinement, overlays, and export.
# =============================================================================

import streamlit as st
from PIL import Image
import os, json, tempfile
from cryptography.fernet import Fernet
from utils_image import trim_whitespace, resize_for_preview
from utils_layout import auto_detect_layout, draw_layout_overlay
from utils_ocr import form_parser_ocr, vision_api_ocr_boxes, documentai_ocr_boxes
from utils_refine import refine_layout_with_zones, export_layout_json
from image_cropper import crop_and_confirm_forms, draw_zone_overlay

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
manual_split = st.sidebar.checkbox("Enable Manual Zone Splitting", value=False)

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
        # === Save profile as downloadable JSON ===
        profile_json = json.dumps(profile, indent=2)
        st.sidebar.download_button(
            label="💾 Export Profile JSON",
            data=profile_json,
            file_name=f"{selected_profile}_profile.json",
            mime="application/json"
        )
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

            # === OCR Box Extraction
            if use_docai and docai_config:
                response = form_parser_ocr(clean, **docai_config)
                boxes, _ = documentai_ocr_boxes(response, clean)
            else:
                boxes = vision_api_ocr_boxes(clean)

            # === Auto Layout Detection
            layout = auto_detect_layout(clean, use_docai=use_docai, config=docai_config)

            # === Refine Layout Zones
            layout, auto_overlay = refine_layout_with_zones(layout, boxes, clean, manual=False, form_id=form_id)

            # === Session key to track manual mode
            manual_mode_key = f"manual_mode_{form_id}"
            if manual_mode_key not in st.session_state:
                st.session_state[manual_mode_key] = False

            # === Toggle between auto and manual layout
            toggle = st.radio(
                "🧭 Layout Mode",
                ["Auto", "Manual"],
                index=0 if not st.session_state[manual_mode_key] else 1,
                key=f"layout_toggle_{form_id}"
            )

            if toggle == "Auto":
                st.session_state[manual_mode_key] = False
                st.image(resize_for_preview(auto_overlay), caption="🧠 Auto Layout Overlay", use_column_width=True)

            else:
                st.session_state[manual_mode_key] = True
                st.markdown("### 🔧 Manual Layout Editor")

            def slider_box(label, default):
                x1 = st.slider(f"{label} x1", 0.0, 1.0, default[0], 0.01, key=f"{form_id}_{label}_x1")
                y1 = st.slider(f"{label} y1", 0.0, 1.0, default[1], 0.01, key=f"{form_id}_{label}_y1")
                x2 = st.slider(f"{label} x2", x1 + 0.01, 1.0, default[2], 0.01, key=f"{form_id}_{label}_x2")
                y2 = st.slider(f"{label} y2", y1 + 0.01, 1.0, default[3], 0.01, key=f"{form_id}_{label}_y2")
                return [x1, y1, x2, y2]

            layout["master_box"] = slider_box("Master", layout.get("master_box", [0.0, 0.0, 1.0, 0.5]))
            layout["group_a_box"] = slider_box("Group A", layout.get("group_a_box", [0.0, 0.0, 1.0, 0.25]))
            layout["group_b_box"] = slider_box("Group B", layout.get("group_b_box", [0.0, 0.25, 1.0, 0.5]))
            layout["detail_box"] = slider_box("Detail", layout.get("detail_box", [0.0, 0.5, 1.0, 1.0]))
            layout["detail_top_box"] = slider_box("Detail Top", layout.get("detail_top_box", [0.0, 0.5, 1.0, 0.75]))
            layout["detail_bottom_box"] = slider_box("Detail Bottom", layout.get("detail_bottom_box", [0.0, 0.75, 1.0, 1.0]))

            manual_overlay = draw_layout_overlay(clean.copy(), layout)
            st.image(resize_for_preview(manual_overlay), caption="🖍️ Manual Layout Overlay", use_column_width=True)

            # === Final overlay with toggles
            draw_zone_overlay(clean, layout, form_id)

            # === Export layout
            export_layout_json(layout, form_id)
