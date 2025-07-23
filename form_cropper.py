# =============================================================================
# FILE: form_cropper.py
# VERSION: 1.0.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Interactive cropping tool for scanned registry forms.
#              Supports canvas-based selection, preview, confirmation,
#              undo, and persistent crop history across reruns.
# =============================================================================

from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from utils_image import resize_for_preview

def crop_and_confirm_forms(image: Image.Image, max_crops=5, key_prefix="crop"):
    """
    Interactive cropping tool for extracting form regions from a scanned image.
    Supports preview, confirmation, and undo.
    """
    st.markdown("### ✂️ Crop Forms")
    confirmed_forms = []
    canvas_height = image.height
    canvas_width = image.width

    if f"{key_prefix}_history" not in st.session_state:
        st.session_state[f"{key_prefix}_history"] = []

    for i in range(max_crops):
        crop_key = f"{key_prefix}_{i}"
        st.markdown(f"#### Crop Form {i+1}")

        canvas_result = st_canvas(
            background_image=image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            key=f"{crop_key}_canvas"
        )

        preview_crop = None
        if canvas_result.json_data and "objects" in canvas_result.json_data:
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    x1 = int(obj["left"])
                    y1 = int(obj["top"])
                    x2 = int(obj["left"] + obj["width"])
                    y2 = int(obj["top"] + obj["height"])
                    preview_crop = image.crop((x1, y1, x2, y2))
                    st.image(resize_for_preview(preview_crop), caption=f"🖼️ Preview Crop {i+1}", use_column_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"✅ Confirm Crop {i+1}", key=f"{crop_key}_confirm") and preview_crop:
                st.session_state[f"{key_prefix}_history"].append(preview_crop)
                st.success(f"Form {i+1} confirmed.")
        with col2:
            if st.button(f"↩️ Undo Last", key=f"{crop_key}_undo"):
                if st.session_state[f"{key_prefix}_history"]:
                    st.session_state[f"{key_prefix}_history"].pop()
                    st.warning("Last crop removed.")

    confirmed_forms = st.session_state[f"{key_prefix}_history"]
    return confirmed_forms
