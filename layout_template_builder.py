import streamlit as st
from PIL import Image, ImageDraw
import json
from io import BytesIO
from difflib import get_close_matches, SequenceMatcher
from streamlit_image_coordinates import streamlit_image_coordinates

FIELD_LABELS = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# Session initialization
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {}
if "click_points" not in st.session_state:
    st.session_state.click_points = []
if "images" not in st.session_state:
    st.session_state.images = {}

st.set_page_config(layout="wide", page_title="🧩 OCR Template Builder")
st.title("🧩 OCR Layout Template Builder")

# === 1. File Management & Batch Upload ===
uploaded_files = st.file_uploader(
    "📁 Upload scanned form(s)",
    type=["jpg", "jpeg", "png", "jp2"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        st.session_state.images[file.name] = image
        st.session_state.form_layouts[file.name] = {}
    st.success(f"✅ Loaded {len(uploaded_files)} form(s)")

# === 2. Image Canvas & Box Tagging ===
selected_image = st.selectbox("🖼️ Select form to tag", list(st.session_state.images.keys()) or ["(none)"])
selected_label = st.selectbox("🏷️ Select label", FIELD_LABELS)

if selected_image and selected_image != "(none)":
    image = st.session_state.images[selected_image]
    layout = st.session_state.form_layouts[selected_image]

    st.markdown("### 👆 Click twice to define a field box")
    coords = streamlit_image_coordinates(image)
    if coords:
        st.session_state.click_points.append((coords["x"], coords["y"]))
        st.toast(f"Point {len(st.session_state.click_points)}: ({coords['x']}, {coords['y']})")

        if len(st.session_state.click_points) == 2:
            (x1, y1), (x2, y2) = st.session_state.click_points
            st.session_state.click_points = []

            box = {
                "x1": min(x1, x2), "y1": min(y1, y2),
                "x2": max(x1, x2), "y2": max(y1, y2)
            }

            match = get_close_matches(selected_label, FIELD_LABELS, n=1)
            suggested = match[0] if match else selected_label
            score = round(SequenceMatcher(None, selected_label, suggested).ratio() * 100, 2)

            layout[suggested] = box
            st.session_state.form_layouts[selected_image] = layout
            st.success(f"✅ Tagged '{suggested}' (Confidence: {score}%) in {selected_image}")

    # === Overlay Drawing Preview ===
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for label, box in layout.items():
        draw.rectangle([(box["x1"], box["y1"]), (box["x2"], box["y2"])], outline="green", width=3)
        draw.text((box["x1"], box["y1"] - 12), label, fill="green")
    st.image(preview, caption=f"Tagged layout for {selected_image}", use_column_width=True)

# === 3. Auto-Alignment (TODO) ===
# def align_layout_to_reference(image, layout):
#     # TODO: Add logic to detect header region and shift all boxes
#     return aligned_layout

# === 4. Completeness Tracker ===
st.markdown("## 📊 Form Layout Completeness")
for fname, fields in st.session_state.form_layouts.items():
    count = len(fields)
    status = "🟢 Complete" if count == 8 else "🟡 Partial" if count >= 5 else "🔴 Incomplete"
    st.markdown(f"{status} — `{fname}`: {count} fields")

# === 5. Smart Label Suggestion (TODO) ===
# def suggest_label_from_context(text_snippet):
#     # TODO: Use OCR context or tagging history to predict best label
#     return label

# === Export Layouts ===
st.markdown("## 💾 Export Tagged Layouts")
st.download_button(
    label="💾 Download form_layouts.json",
    data=json.dumps(st.session_state.form_layouts, indent=2, ensure_ascii=False),
    file_name="form_layouts.json",
    mime="application/json"
)
