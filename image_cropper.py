# =============================================================================
# FILE: image_cropper.py
# VERSION: 1.3
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Interactive cropping and layout zone visualization.
#              Uses st_cropper for drag-based selection and preview.
# =============================================================================

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_cropper import st_cropper

# === Interactive Cropping ===
def crop_and_confirm_forms(image: Image.Image, max_crops=5) -> list:
    """
    Allows user to interactively crop multiple regions from the image.
    Returns list of confirmed cropped form images.
    """
    st.markdown("### ✂️ Crop Forms from Image")
    confirmed = []

    for i in range(max_crops):
        st.markdown(f"#### Crop Region {i+1}")

        cropped_img = st_cropper(
            image,
            box_color="blue",
            aspect_ratio=None,
            return_type="image",
            key=f"crop_{i}",
            realtime_update=True
        )

        # === Live preview of cropped region
        st.image(cropped_img, caption=f"🔍 Preview Crop {i+1}", use_column_width=True)
        # === Confirm crop

        if st.button(f"✅ Confirm Crop {i+1}"):
            confirmed.append(cropped_img)
            st.image(cropped_img, caption=f"🧾 Cropped Form {i+1}", use_column_width=True)

    if not confirmed:
        st.info("ℹ️ No crops confirmed yet.")
    return confirmed

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
