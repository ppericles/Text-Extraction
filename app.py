import streamlit as st
from PIL import Image, ImageOps
from google.cloud import vision
from google.oauth2 import service_account
import io

# Load credentials from Streamlit secrets
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = vision.ImageAnnotatorClient(credentials=credentials)

# Optional: Greek character correction
def correct_greek_misreads(text):
    substitutions = {
        "P": "Ρ", "H": "Η", "N": "Ν", "A": "Α", "B": "Β",
        "E": "Ε", "T": "Τ", "X": "Χ", "Y": "Υ", "M": "Μ"
    }
    for latin, greek in substitutions.items():
        text = text.replace(latin, greek)
    return text

# Preprocess image for better OCR
def preprocess_image(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.resize((1024, int(1024 * image.height / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

# OCR function using Google Vision
def extract_text_google_vision(uploaded_file):
    content = preprocess_image(uploaded_file)
    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["el"])
    try:
        response = client.text_detection(image=image, image_context=image_context)
        return response.text_annotations[0].description if response.text_annotations else "No text found"
    except Exception as e:
        return f"❌ Vision API error: {e}"

# Streamlit UI
st.title("🧾 Greek Handwriting OCR (Google Vision)")

uploaded_file = st.file_uploader("Upload a handwritten Greek image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    uploaded_file.seek(0)
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)

    raw_text = extract_text_google_vision(uploaded_file)
    corrected_text = correct_greek_misreads(raw_text)

    st.subheader("🔍 Raw OCR Output")
    st.text_area("Raw Text", raw_text, height=150)

    st.subheader("✅ Corrected Greek Text")
    st.text_area("Corrected Text", corrected_text, height=150)
