# =============================================================================
# FILE: image_cropper.py
# VERSION: 1.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Interactive cropping and layout zone visualization.
#              Supports multi-form extraction and layout overlays.
# =============================================================================

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# === Interactive Cropping ===
def crop_and_confirm_forms(image: Image.Image, max_crops=5) -> list:
    """
    Allows user to crop multiple regions from the uploaded image.
    Returns list of confirmed cropped form images.
    """
    st.markdown("### ✂️ Crop Forms from Image")
    w, h = image.size
    crops = []

    for i in range(max_crops):
        st.markdown(f"#### Crop Region {i+1}")
        x1 = st.slider(f"x1 [{i+1}]", 0, w - 1, 0, 1)
        y1 = st.slider(f"y1 [{i+1}]", 0, h - 1, 0, 1)
        x2 = st.slider(f"x2 [{i+1}]", x1 + 1, w, w, 1)
        y2 = st.slider(f"y2 [{i+1}]", y1 + 1, h, h, 1)

        if st.button(f"✅ Confirm Crop {i+1}"):
            cropped = image.crop((x1, y1, x2, y2))
            crops.append(cropped)
            st.image(cropped, caption=f"🧾 Cropped Form {i+1}", use_column_width=True)

    if not crops:
        st.warning("⚠️ No crops confirmed yet.")
    return crops

# === Layout Zone Overlay ===
def draw_zone_overlay(image: Image.Image, layout: dict, form_id: str):
    """
    Draws layout zones on the image and displays it in Streamlit.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    w, h = image.size

    color_map = {
        "master_box": "green",
        "detail_box": "gray",
        "group_a_box": "blue",
        "group_b_box": "red",
        "detail_top_box": "purple",
        "detail_bottom_box": "orange"
    }

    for label, color in color_map.items():
        box = layout.get(label)
        if box and len(box) == 4:
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, (w, h, w, h))]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 5, y1 + 5), label.replace("_box", "").upper(), fill=color, font=font)

    st.image(overlay, caption=f"🖍️ Final Layout Overlay — `{form_id}`", use_column_width=True)
