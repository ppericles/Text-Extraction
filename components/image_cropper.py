# ==== components/image_cropper.py ====

import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

def crop_and_confirm_forms(image, max_crops=5):
    """
    Crop multiple forms from one image with confirmation.

    Args:
        image (PIL.Image): The scanned image
        max_crops (int): Max crops to offer

    Returns:
        list[PIL.Image]: Confirmed cropped images
    """
    assert isinstance(image, Image.Image), "❌ Input image must be a PIL.Image"

    st.markdown("## ✂️ Manual Cropping")
    st.image(image, caption="🖼️ Original Image", use_column_width=True)

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

        assert isinstance(cropped_img, Image.Image), f"❌ Crop #{i + 1} is not a valid PIL.Image"

        st.image(cropped_img, caption=f"Crop #{i + 1}", use_column_width=True)
        st.json(crop_box, expanded=False)

        confirm = st.checkbox(f"✅ Keep Crop #{i + 1}", value=True, key=f"confirm_{i}")
        if confirm:
            crops.append(cropped_img)

    # 🔍 Final confirmation grid
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
