# ==== components/image_cropper.py ====

import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

def crop_and_confirm_forms(image, max_crops=5):
    """
    Crop multiple forms from one image, with optional rotation and confirmation.

    Args:
        image (PIL.Image): The scanned image
        max_crops (int): Max crops to offer

    Returns:
        list[dict]: Confirmed cropped forms with image and rotation metadata
    """
    st.markdown("## ✂️ Manual Cropping")

    # 🔄 Rotation control
    angle = st.slider("🔄 Rotate Image (degrees)", -180, 180, step=90, value=0)
    rotated_image = image.rotate(angle, expand=True)
    st.image(rotated_image, caption="🖼️ Rotated Image", use_column_width=True)

    crops = []

    for i in range(max_crops):
        st.markdown(f"### 🖼️ Crop #{i + 1}")
        cropped_img, crop_box = st_cropper(
            rotated_image,
            return_type="both",
            box_color="orange",
            aspect_ratio=None,
            realtime_update=True,
            key=f"crop_{i}"
        )
        st.image(cropped_img, caption=f"Crop #{i + 1}", use_column_width=True)
        st.json(crop_box, expanded=False)

        confirm = st.checkbox(f"✅ Keep Crop #{i + 1}", value=True, key=f"confirm_{i}")
        if confirm:
            crops.append({
                "image": cropped_img,
                "box": crop_box,
                "angle": angle  # ✅ Store rotation for later correction
            })

    # 🔍 Final confirmation grid
    st.markdown("## 🔍 Confirm Final Selection")
    final = []
    cols = st.columns(3)
    for idx, form in enumerate(crops):
        with cols[idx % 3]:
            st.image(form["image"], caption=f"Form {idx + 1}", use_column_width=True)
            keep = st.checkbox(f"Keep Form {idx + 1}", value=True, key=f"final_confirm_{idx}")
            if keep:
                final.append(form)

    return final
