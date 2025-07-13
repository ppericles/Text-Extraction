import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("🖱️ Click Tracker Test")

uploaded_file = st.file_uploader("📎 Upload image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    coords = streamlit_image_coordinates(image)
    if coords:
        st.write(f"You clicked at: ({coords['x']}, {coords['y']})")
