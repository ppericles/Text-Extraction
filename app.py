import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.title("🧠 Greek Handwriting OCR (TrOCR Debug Mode)")
st.caption("Upload an image with Greek handwritten text and see what TrOCR predicts.")

# --- Try loading TrOCR model ---
try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    st.success("✅ TrOCR model loaded")
except Exception as e:
    st.error(f"❌ Failed to load TrOCR model: {e}")
    st.stop()

# --- Upload image ---
uploaded_file = st.file_uploader("📎 Upload handwritten image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        resized = image.resize((384, 384))
        st.image(resized, caption="📷 Resized Image (384×384)", use_column_width=True)

        with st.spinner("🔍 Running TrOCR..."):
            inputs = processor(images=resized, return_tensors="pt")
            generated_ids = model.generate(inputs.pixel_values)
            prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        st.success("✅ Prediction complete")
        st.subheader("📝 Predicted Text")
        st.text(prediction)

        print("🔍 [DEBUG] TrOCR Output:", prediction)

    except Exception as e:
        st.error(f"❌ OCR failed: {e}")
        print("🛠️ [DEBUG] Error during prediction:", e)
