# ==== image_cropper.py ====

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

def crop_image_ui(image, label="✂️ Crop Registry Scan", box_color="red"):
    """
    Display interactive cropper in Streamlit and return cropped image.

    Args:
        image (PIL.Image): Original uploaded image
        label (str): UI label above cropper
        box_color (str): Rectangle color

    Returns:
        PIL.Image: Cropped image
    """
    st.subheader(label)
    cropped_img = st_cropper(
        image,
        realtime_update=True,
        box_color=box_color,
        aspect_ratio=None,
        return_type="PIL"
    )
    st.image(cropped_img, caption="🖼️ Cropped Image Ready for Parsing", use_column_width=True)
    return cropped_img

def batch_crop_images_ui(uploaded_files, box_color="red"):
    """
    Crop multiple uploaded images interactively.

    Args:
        uploaded_files (list): List of uploaded file objects
        box_color (str): Rectangle color

    Returns:
        dict: Mapping of filename → cropped PIL image
    """
    cropped_results = {}

    for idx, file in enumerate(uploaded_files, start=1):
        st.markdown(f"---\n### ✂️ Crop Image {idx}: `{file.name}`")
        image = Image.open(file).convert("RGB")

        cropped = st_cropper(
            image,
            realtime_update=True,
            box_color=box_color,
            aspect_ratio=None,
            return_type="PIL",
            key=f"crop_{idx}"
        )

        st.image(cropped, caption=f"🖼️ Cropped Image {idx}", use_column_width=True)
        cropped_results[file.name] = cropped

    return cropped_results
