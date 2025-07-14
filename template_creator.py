import streamlit as st
from PIL import Image, ImageDraw
import json
from difflib import get_close_matches, SequenceMatcher
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO

FIELD_LABELS = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

FORM_IDS = [1, 2, 3]

st.set_page_config(layout="wide", page_title="OCR Template Creator")
st.title("📐 OCR Template Creator")

# Session State Setup
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in FORM_IDS}
if "click_points" not in st.session_state:
    st.session_state.click_points = []

form_num = st.sidebar.selectbox("📄 Select Φόρμα", FORM_IDS)
selected_label = st.sidebar.selectbox("🏷️ Assign Label", FIELD_LABELS)

uploaded_file = st.file_uploader("📤 Upload scanned form", type=["jpg", "jpeg", "png", "jp2"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

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

            st.session_state.form_layouts[form_num][suggested] = box
            st.success(f"✅ Tagged '{suggested}' (Confidence: {score}%) in Φόρμα {form_num}")

    # Box Preview on Image
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for label, box in st.session_state.form_layouts[form_num].items():
        draw.rectangle([(box["x1"], box["y1"]), (box["x2"], box["y2"])], outline="green", width=3)
        draw.text((box["x1"], box["y1"] - 12), label, fill="green")
    st.image(preview, caption=f"Preview for Φόρμα {form_num}", use_column_width=True)

# Layout Summary
st.markdown("## 📊 Φόρμα Completeness")
for fid in FORM_IDS:
    count = len(st.session_state.form_layouts.get(fid, {}))
    if count == len(FIELD_LABELS):
        status = "🟢 Complete"
    elif count >= 5:
        status = "🟡 Partial"
    else:
        status = "🔴 Incomplete"
    st.write(f"{status} — Φόρμα {fid}: {count} fields")

# Export
st.download_button(
    label="💾 Export Layout as JSON",
    data=json.dumps(st.session_state.form_layouts, indent=2, ensure_ascii=False),
    file_name="form_layouts.json",
    mime="application/json"
)
