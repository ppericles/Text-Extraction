import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import json
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2

# --- App setup ---
st.set_page_config(layout="wide", page_title="Greek Handwriting OCR Form Parser")
st.title("🇬🇷 EasyOCR-Powered Form Parser with Field Annotation")

field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# --- Initialize session ---
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in [1, 2, 3]}
if "click_stage" not in st.session_state:
    st.session_state.click_stage = "start"
if "ocr_blocks" not in st.session_state:
    st.session_state.ocr_blocks = []

# --- Sidebar config ---
form_num = st.sidebar.selectbox("📄 Φόρμα", [1, 2, 3])
field_label = st.sidebar.selectbox("📝 Field Name", field_labels)

# --- Import layout ---
layout_file = st.sidebar.file_uploader("📂 Import layout (.json)", type=["json"])
if layout_file:
    try:
        st.session_state.form_layouts = json.load(layout_file)
        st.success("✅ Layout imported")
    except Exception as e:
        st.error(f"Layout import failed: {e}")

# --- Upload form image ---
uploaded_file = st.file_uploader("📎 Upload scanned handwritten form", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    if max(image.size) > 1800:
        image = image.resize((min(image.width, 1800), min(image.height, 1800)))
    img_array = np.array(image)

    st.image(image, caption="📷 Uploaded Image", use_container_width=True)
    st.caption("👆 Click to define bounding boxes: top-left first, then bottom-right.")

    coords = streamlit_image_coordinates(image, key="coord_click")
    if coords:
        x, y = coords["x"], coords["y"]
        field_boxes = st.session_state.form_layouts[form_num]

        if st.session_state.click_stage == "start":
            field_boxes[field_label] = {"x1": x, "y1": y}
            st.session_state.click_stage = "end"
            st.info(f"🟩 Top-left corner set for '{field_label}'. Click bottom-right.")
        else:
            if field_label not in field_boxes:
                field_boxes[field_label] = {}
            field_boxes[field_label].update({"x2": x, "y2": y})
            st.session_state.click_stage = "start"
            st.success(f"✅ Box saved for '{field_label}' in Φόρμα {form_num}.")

    # --- OCR with EasyOCR
    reader = easyocr.Reader(['el'])
    results = reader.readtext(img_array)
    st.session_state.ocr_blocks = [
        {
            "text": text,
            "confidence": float(conf),
            "center": (np.mean([pt[0] for pt in bbox]), np.mean([pt[1] for pt in bbox]))
        }
        for bbox, text, conf in results
    ]

    # --- Display results by form
    st.subheader("🧠 OCR Field Extraction")
    for i in [1, 2, 3]:
        st.markdown(f"### 📄 Φόρμα {i}")
        layout = st.session_state.form_layouts[i]
        for label in field_labels:
            box = layout.get(label)
            if box and all(k in box for k in ["x1", "y1", "x2", "y2"]):
                xmin, xmax = sorted([box["x1"], box["x2"]])
                ymin, ymax = sorted([box["y1"], box["y2"]])
                match = next(
                    (b for b in st.session_state.ocr_blocks
                     if xmin <= b["center"][0] <= xmax and ymin <= b["center"][1] <= ymax),
                    None
                )
                val = match["text"] if match else "(no match)"
                st.text_input(label, val, key=f"{i}_{label}")

    # --- Optional visual overlay
    if st.checkbox("🖼️ Show detected text boxes"):
        img_overlay = img_array.copy()
        for b in results:
            bbox, text, conf = b
            pts = np.array(bbox, dtype=np.int32)
            img_overlay = cv2.polylines(img_overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            img_overlay = cv2.putText(img_overlay, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, (255, 0, 0), 1)
        st.image(img_overlay, caption="🖼️ Text Box Overlay", use_container_width=True)

# --- Layout export
st.download_button(
    label="💾 Export Layout as JSON",
    data=json.dumps(st.session_state.form_layouts, ensure_ascii=False, indent=2),
    file_name="form_layouts.json",
    mime="application/json"
)
