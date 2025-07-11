import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load TrOCR base handwritten model
st.title("🧠 Greek Handwriting OCR (TrOCR Test App)")
st.caption("Upload an image with Greek handwritten text and see what TrOCR predicts.")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Upload image
uploaded_file = st.file_uploader("📎 Upload handwritten image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize((384, 384))
    st.image(resized, caption="📷 Resized Image (384×384)", use_column_width=True)

    # Run TrOCR
    with st.spinner("🔍 Running TrOCR model..."):
        inputs = processor(images=resized, return_tensors="pt")
        generated_ids = model.generate(inputs.pixel_values)
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.subheader("📝 Predicted Text")
    st.text(prediction)
