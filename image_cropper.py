# ==== image_cropper.py ====

from PIL import Image
from streamlit_cropper import st_cropper
import streamlit as st

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
