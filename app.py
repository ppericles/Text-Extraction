import streamlit as st
from PIL import Image
import numpy as np
import torch
import json
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from streamlit_image_coordinates import streamlit_image_coordinates

# Load pre-trained TrOCR model (base version)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# --- App Setup ---
st.set_page_config(layout="wide", page_title="Greek Handwritten TrOCR Parser")
st.title("🇬🇷 TrOCR Streamlit OCR with Field Annotation")

field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# --- State Initialization ---
if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: {} for i in [1, 2, 3]}
if "click_stage" not in st.session_state:
    st.session_state.click_stage = "start"

form_num = st.sidebar.selectbox("📄 Φόρμα", [1, 2, 3])
field_label = st.sidebar.selectbox("📝 Field Name", field_labels)

layout_file = st.sidebar.file_uploader("📂 Import layout (.json)", type=["json"])
if layout_file:
    try:
        st.session_state.form_layouts = json.load(layout_file)
        st.success("✅ Layout imported")
    except Exception as e:
        st.error(f"Import failed: {e}")

# --- Upload Form Image ---
uploaded_file = st.file_uploader("📎 Upload scanned handwritten form", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    if max(image.size) > 1800:
        image = image.resize((min(image.width, 1800), min(image.height, 1800)))
    img_array = np.array(image)

    st.image(image, caption="📷 Uploaded Image", use_container_width=True)
    coords = streamlit_image_coordinates(image, key="coord_click")

    # --- Box Click Logic ---
    if coords:
        x, y = coords["x"], coords["y"]
        field_boxes = st.session_state.form_layouts[form_num]
        if st.session_state.click_stage == "start":
            field_boxes[field_label] = {"x1": x, "y1": y}
            st.session_state.click_stage = "end"
            st.info(f"🟩 Top-left set for '{field_label}'. Click bottom-right.")
        else:
            field_boxes[field_label].update({"x2": x, "y2": y})
            st.session_state.click_stage = "start"
            st.success(f"✅ Box saved for '{field_label}' in Φόρμα {form_num}.")

    # --- Field-Level TrOCR Extraction ---
    st.subheader("🧠 OCR Field Extraction (TrOCR)")
    for i in [1, 2, 3]:
        st.markdown(f"### 📄 Φόρμα {i}")
        layout = st.session_state.form_layouts[i]
        for label in field_labels:
            box = layout.get(label)
            if box and all(k in box for k in ["x1", "y1", "x2", "y2"]):
                xmin, xmax = sorted([box["x1"], box["x2"]])
                ymin, ymax = sorted([box["y1"], box["y2"]])
                crop_img = img_array[ymin:ymax, xmin:xmax]
                if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                    pil_crop = Image.fromarray(crop_img).resize((384, 384))
                    pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values
                    with torch.no_grad():
                        generated_ids = model.generate(pixel_values)
                    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    st.text_input(label, prediction, key=f"{i}_{label}")
                else:
                    st.text_input(label, "(invalid crop)", key=f"{i}_{label}")

    # --- Optional Visual Overlay ---
    if st.checkbox("🖼️ Show annotated boxes"):
        overlay = img_array.copy()
        for i in [1, 2, 3]:
            layout = st.session_state.form_layouts[i]
            for label in field_labels:
                box = layout.get(label)
                if box and all(k in box for k in ["x1", "y1", "x2", "y2"]):
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(overlay, f"{label} (Φ{i})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        st.image(overlay, caption="🖼️ Field Boxes", use_container_width=True)

    # --- Export Layout
    st.download_button(
        label="💾 Export Layout as JSON",
        data=json.dumps(st.session_state.form_layouts, ensure_ascii=False, indent=2),
        file_name="form_layouts.json",
        mime="application/json"
    )
