# ==== components/image_cropper.py ====

import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

def crop_and_confirm_forms(image, max_crops=5):
    """
    Manually crop multiple forms from one scanned image, then confirm selections.

    Args:
        image (PIL.Image): The scanned image containing stacked forms
        max_crops (int): Maximum number of crops allowed

    Returns:
        list[PIL.Image]: List of confirmed cropped form images
    """
    st.markdown("## ✂️ Manual Cropping")
    crops = []

    for i in range(max_crops):
        st.markdown(f"### 🖼️ Crop #{i + 1}")
        cropped_img, crop_box = st_cropper(
            image,
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
            crops.append(cropped_img)

    # Final confirmation grid
    st.markdown("## 🔍 Confirm Final Selection")
    final = []
    cols = st.columns(3)
    for idx, img in enumerate(crops):
        with cols[idx % 3]:
            st.image(img, caption=f"Form {idx + 1}", use_column_width=True)
            keep = st.checkbox(f"Keep Form {idx + 1}", value=True, key=f"final_confirm_{idx}")
            if keep:
                final.append(img)

    return final
