import streamlit as st
from PIL import Image, ImageDraw
import json
import os
from io import BytesIO
from difflib import get_close_matches
import pandas as pd
from google.cloud import vision
from streamlit_image_coordinates import streamlit_image_coordinates

# Optional: replace with your own utility modules
from utils.image_utils import image_to_base64
from utils.ocr_utils import normalize, detect_header_regions, compute_form_bounds
from utils.layout_utils import get_form_bounding_box

st.set_page_config(layout="wide", page_title="Greek OCR Annotator")
st.title("🇬🇷 Greek OCR Annotator")

# === 🧠 Session Initialization ===
field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]
form_ids = [1, 2, 3]

for key, default in {
    "form_layouts": {i: {} for i in form_ids},
    "ocr_blocks": [],
    "click_points": [],
    "extracted_values": {i: {} for i in form_ids},
    "resolved_forms": set(),
    "current_low_index": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# === 🗂️ Sidebar Controls ===
view_mode = st.sidebar.radio("🧭 View Mode", ["Tagging", "Compare All Forms"], index=0)
form_num = st.sidebar.selectbox("📄 Φόρμα", form_ids, index=form_ids.index(st.session_state.get("form_num", form_ids[0])))
selected_label = st.sidebar.selectbox("🏷️ Field Label", field_labels)

cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

layout_file = st.sidebar.file_uploader("📂 Import layout (.json)", type=["json"])
if layout_file:
    raw_layout = json.load(layout_file)
    imported = {int(k): v for k, v in raw_layout.items() if k.isdigit()}
    st.session_state.form_layouts.update(imported)
    st.sidebar.success("✅ Layout imported")

uploaded_file = st.file_uploader("📎 Upload scanned form", type=["jpg", "jpeg", "png", "jp2"])

# === 📶 Sidebar Confidence Summary ===
st.sidebar.markdown("### 🧠 Confidence Summary per Φόρμα")
for fid in form_ids:
    layout = st.session_state.form_layouts.get(fid, {})
    blocks = st.session_state.ocr_blocks
    matched = 0
    for label in field_labels:
        box = layout.get(label)
        if box and blocks:
            hits = [b for b in blocks if box["x1"] <= b["center"][0] <= box["x2"]
                                  and box["y1"] <= b["center"][1] <= box["y2"]]
            if hits:
                matched += 1
    pct = round((matched / len(field_labels)) * 100, 1)
    color = "🟢" if pct >= 85 else "🟡" if pct >= 50 else "🔴"
    st.sidebar.write(f"{color} Φόρμα {fid}: {pct}% matched")

# === 🖼️ Image Preview ===
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    layout = st.session_state.form_layouts.get(form_num, {})
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for label, box in layout.items():
        draw.rectangle([(box["x1"], box["y1"]), (box["x2"], box["y2"])], outline="green", width=2)
        draw.text((box["x1"], box["y1"] - 10), label, fill="green")
    st.image(preview, caption=f"Φόρμα {form_num} Layout Preview", use_container_width=True)

# === 🖱️ Tagging Interaction + Auto Extract ===
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
            tag = get_close_matches(selected_label, field_labels, n=1)
            label = tag[0] if tag else selected_label
            st.session_state.form_layouts.setdefault(form_num, {})[label] = box
            st.success(f"✅ Saved '{label}' for Φόρμα {form_num}")

            # 🧠 Live re-extract after tagging
            blocks = st.session_state.ocr_blocks
            hits = [b["text"] for b in blocks if box["x1"] <= b["center"][0] <= box["x2"]
                                          and box["y1"] <= b["center"][1] <= box["y2"]]
            value = " ".join(hits)
            st.session_state.extracted_values[form_num][label] = value
            st.toast(f"🔁 Updated extracted value for '{label}'")

import layout_detector  # if not already imported

if uploaded_file and cred_file and st.button("🔍 Auto-Detect Layout with AI"):
    temp_path = "temp_scan.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    detected_layout = layout_detector.detect_layout_and_extract_fields(temp_path)
    st.session_state.form_layouts.update(detected_layout)
    st.success("✅ Layout loaded from LayoutParser")

# === 🔍 OCR Trigger
if uploaded_file and cred_file and st.button("🔍 Run OCR"):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    content = buffer.getvalue()
    client = vision.ImageAnnotatorClient()
    response = client.document_text_detection(image=vision.Image(content=content))
    annotations = response.text_annotations[1:]
    blocks = []
    for a in annotations:
        v = a.bounding_poly.vertices
        xs = [v.x for v in v if v.x is not None]
        ys = [v.y for v in v if v.y is not None]
        if xs and ys:
            center = (sum(xs) // len(xs), sum(ys) // len(ys))
            blocks.append({"text": a.description, "center": center})
    st.session_state.ocr_blocks = blocks
    st.success("✅ OCR completed")

# === 🩺 Field Diagnostics
if view_mode == "Tagging":
    st.subheader(f"🧠 Extracted Fields — Φόρμα {form_num}")
    layout = st.session_state.form_layouts.get(form_num, {})
    blocks = st.session_state.ocr_blocks
    extracted = {}

    with st.expander("🩺 Field Diagnostics", expanded=True):
        for label in field_labels:
            box = layout.get(label)
            match_count, value = 0, ""
            if box and blocks:
                hits = [b["text"] for b in blocks if box["x1"] <= b["center"][0] <= box["x2"]
                                                and box["y1"] <= b["center"][1] <= box["y2"]]
                value = " ".join(hits)
                match_count = len(hits)
            extracted[label] = value
            status = "🟢" if match_count else "🔴"
            st.text_input(f"{status} {label} — {match_count} hit(s)", value, key=f"{form_num}_{label}")

    st.session_state.extracted_values[form_num] = extracted

# === 📊 Resolution Navigator + Export
low_conf_forms = []
dashboard_data = []

for fid in form_ids:
    layout = st.session_state.form_layouts.get(fid, {})
    blocks = st.session_state.ocr_blocks
    matched = 0
    for label in field_labels:
        box = layout.get(label)
        if box and blocks:
            hits = [b for b in blocks if box["x1"] <= b["center"][0] <= box["x2"]
                                  and box["y1"] <= b["center"][1] <= box["y2"]]
            if hits:
                matched += 1
    pct = round((matched / len(field_labels)) * 100, 1)
    status = "✅ Resolved" if fid in st.session_state.resolved_forms else "❌ Pending"
    if pct < 75 and fid not in st.session_state.resolved_forms:
        low_conf_forms.append(fid)
    dashboard_data.append({"Φόρμα": f"Φόρμα {fid}", "✅ Matched %": pct, "🔄 Status": status})

resolved = len(st.session_state.resolved_forms)
remaining = len(low_conf_forms)
total = resolved + remaining

st.sidebar.markdown("### 📍 Resolution Progress")
st.sidebar.progress(resolved / total if total else 1.0)
st.sidebar.write(f"✅ Resolved: {resolved} / {total}")

queue = [fid for fid in low_conf_forms if fid not in st.session_state.resolved_forms]
if queue:
    if st.button(f"🧭 Next Trouble Spot ({len(queue)} remaining)"):
        next_fid = queue[st.session_state.current_low_index % len(queue)]
        st.session_state.form_num = next_fid
        st.session_state.view_mode = "Tagging"
        st.toast(f"Jumped to Φόρμα {next_fid}")
        st.session_state.current_low_index += 1

if form_num in low_conf_forms and form_num not in st.session_state.resolved_forms:
    if st.button("✅ Mark Φόρμα as Resolved"):
        st.session_state.resolved_forms.add(form_num)
        st.success(f"Φόρμα {form_num} marked as resolved.")

# === 💾 Layout Export ===
st.markdown("## 💾 Export Tagged Layouts")
st.download_button(
    label="💾 Export as JSON",
    data=json.dumps(st.session_state.form_layouts, indent=2, ensure_ascii=False),
    file_name="form_layouts.json",
    mime="application/json"
)
