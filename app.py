# === 🚀 Imports & Setup ===
import streamlit as st
from PIL import Image, ImageDraw
import json
import os
from io import BytesIO
from difflib import get_close_matches, SequenceMatcher
from google.cloud import vision
from streamlit_image_coordinates import streamlit_image_coordinates

from utils.image_utils import image_to_base64
from utils.ocr_utils import normalize, detect_header_regions, compute_form_bounds
from utils.layout_utils import get_form_bounding_box

st.set_page_config(layout="wide", page_title="Greek OCR Annotator")
st.title("🇬🇷 Greek OCR Annotator")

# === 🧠 Session State ===
field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]
form_ids = [1, 2, 3]

if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in form_ids}
if "ocr_blocks" not in st.session_state:
    st.session_state.ocr_blocks = []
if "click_points" not in st.session_state:
    st.session_state.click_points = []
if "extracted_values" not in st.session_state:
    st.session_state.extracted_values = {i: {} for i in form_ids}

# === 🗂️ Sidebar Controls ===
view_mode = st.sidebar.radio("🧭 View Mode", ["Tagging", "Compare All Forms"])
form_num = st.sidebar.selectbox("📄 Φόρμα", form_ids)
selected_label = st.sidebar.selectbox("📝 Field Label", field_labels)

cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

layout_file = st.sidebar.file_uploader("📂 Import layout (.json)", type=["json"])
if layout_file:
    try:
        raw_layout = json.load(layout_file)
        filtered = {int(k): v for k, v in raw_layout.items() if k.isdigit()}
        st.session_state.form_layouts.update(filtered)
        st.sidebar.success("✅ Layout imported")
        for fid, fields in filtered.items():
            count = len(fields)
            status = "🟢" if count == 8 else "🟡" if count >= 5 else "🔴"
            st.sidebar.write(f"{status} Φόρμα {fid}: {count} fields")
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")

uploaded_file = st.file_uploader("📎 Upload scanned form", type=["jpg", "jpeg", "png", "jp2"])

# === 🖼️ Template Preview on Image ===
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    layout = st.session_state.form_layouts.get(form_num, {})
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for label, box in layout.items():
        draw.rectangle([(box["x1"], box["y1"]), (box["x2"], box["y2"])], outline="green", width=2)
        draw.text((box["x1"], box["y1"] - 10), label, fill="green")
    st.image(preview, caption=f"Φόρμα {form_num} Layout Preview", use_column_width=True)

# === 🏷️ Click-Based Tagging ===
if view_mode == "Tagging" and uploaded_file:
    st.markdown("### 👆 Click twice to tag a field box")
    coords = streamlit_image_coordinates(image)
    if coords:
        st.session_state.click_points.append((coords["x"], coords["y"]))
        st.toast(f"Tagged point {len(st.session_state.click_points)}")

        if len(st.session_state.click_points) == 2:
            (x1, y1), (x2, y2) = st.session_state.click_points
            st.session_state.click_points = []
            box = {
                "x1": min(x1, x2), "y1": min(y1, y2),
                "x2": max(x1, x2), "y2": max(y1, y2)
            }
            match = get_close_matches(selected_label, field_labels, n=1)
            tag = match[0] if match else selected_label
            st.session_state.form_layouts.setdefault(form_num, {})[tag] = box
            st.success(f"✅ Saved '{tag}' for Φόρμα {form_num}")

# === 🔍 OCR Execution ===
if uploaded_file and cred_file and st.button("🔍 Run OCR"):
    try:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        content = buffer.getvalue()

        client = vision.ImageAnnotatorClient()
        response = client.document_text_detection(image=vision.Image(content=content))
        annotations = response.text_annotations[1:]  # skip first full-block

        blocks = []
        for a in annotations:
            vertices = a.bounding_poly.vertices
            xs = [v.x for v in vertices if v.x is not None]
            ys = [v.y for v in vertices if v.y is not None]
            if xs and ys:
                center = (sum(xs) // len(xs), sum(ys) // len(ys))
                blocks.append({"text": a.description, "center": center})
        st.session_state.ocr_blocks = blocks
        st.success("✅ OCR complete")
    except Exception as e:
        st.error(f"OCR failed: {e}")

# === 🧠 Extract Fields per Φόρμα ===
if view_mode == "Tagging":
    st.subheader(f"🧠 Extracted Fields — Φόρμα {form_num}")
    extracted = {}
    blocks = st.session_state.ocr_blocks
    layout = st.session_state.form_layouts.get(form_num, {})

    for label in field_labels:
        box = layout.get(label)
        value = ""
        if box and blocks:
            xmin, xmax = box["x1"], box["x2"]
            ymin, ymax = box["y1"], box["y2"]
            hits = [b["text"] for b in blocks if xmin <= b["center"][0] <= xmax and ymin <= b["center"][1] <= ymax]
            value = " ".join(hits)
        extracted[label] = value
        st.text_input(label, value, key=f"{form_num}_{label}")

    st.session_state.extracted_values[form_num] = extracted

# === 📊 Form Comparison Dashboard ===
if view_mode == "Compare All Forms":
    st.subheader("📊 Comparison Across Forms")
    for fid in form_ids:
        layout = st.session_state.form_layouts.get(fid, {})
        extracted = st.session_state.extracted_values.get(fid, {})
        tagged = list(layout.keys())
        missing = [label for label in field_labels if label not in tagged]
        st.markdown(f"### Φόρμα {fid}")
        st.write(f"✅ Tagged: {len(tagged)} / 8")
        if missing:
            st.write("❌ Missing:", ", ".join(missing))
        for label in field_labels:
            value = extracted.get(label, "(no value)")
            st.text_input(label, value, key=f"compare_{fid}_{label}")

# === 💾 Layout Export ===
st.markdown("## 💾 Export Layouts")
st.download_button(
    label="💾 Export Layout as JSON",
    data=json.dumps(st.session_state.form_layouts, indent=2, ensure_ascii=False),
    file_name="form_layouts.json",
    mime="application/json"
)
