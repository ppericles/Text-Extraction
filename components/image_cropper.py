# ==== components/image_cropper.py ====

import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

def crop_image_ui(uploaded_file, return_type="both"):
    """
    Crop a single image interactively and return image + crop box.

    Args:
        uploaded_file (UploadedFile): Streamlit file object
        return_type (str): "image", "box", or "both"

    Returns:
        tuple: (cropped_image, crop_box)
    """
    st.markdown("### ✂️ Crop Image")
    image = Image.open(uploaded_file)

    cropped_image, crop_box = st_cropper(
        image,
        return_type=return_type,
        box_color="red",
        aspect_ratio=None,
        realtime_update=True
    )

    st.image(cropped_image, caption="Cropped Image", use_column_width=True)
    st.json(crop_box, expanded=False)
    return cropped_image, crop_box


def batch_crop_images_ui(uploaded_files, return_type="both"):
    """
    Crop multiple images interactively and return image + crop box per file.

    Args:
        uploaded_files (list): List of Streamlit UploadedFile objects
        return_type (str): "image", "box", or "both"

    Returns:
        dict: filename → {"image": cropped_image, "box": crop_box}
    """
    st.markdown("## ✂️ Batch Cropping")
    results = {}

    for file in uploaded_files:
        name = file.name
        st.markdown(f"### 🖼️ Cropping `{name}`")
        image = Image.open(file)

        cropped_image, crop_box = st_cropper(
            image,
            return_type=return_type,
            box_color="blue",
            aspect_ratio=None,
            realtime_update=True,
            key=name  # ensures unique widget state per image
        )

        st.image(cropped_image, caption=f"Cropped `{name}`", use_column_width=True)
        st.json(crop_box, expanded=False)

        results[name] = {
            "image": cropped_image,
            "box": crop_box
        }

    return results
