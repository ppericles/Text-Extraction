import streamlit as st
from PIL import Image
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

# OCR function using Google Vision (no preprocessing, no language hint)
def extract_text_google_vision(uploaded_file):
    uploaded_file.seek(0)
    content = uploaded_file.read()
    image = vision.Image(content=content)

    try:
        response = client.document_text_detection(image=image)
        return response.full_text_annotation.text if response.full_text_annotation.text else "No text found"
    except Exception as e:
        return f"❌ Vision API error: {e}"

# Streamlit UI
st.title("🧾 Greek Handwriting OCR (Google Vision AI)")

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
