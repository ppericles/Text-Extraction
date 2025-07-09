import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import io

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load Google Vision credentials from Streamlit secrets
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = vision.ImageAnnotatorClient(credentials=credentials)

# Load Greek sentiment analysis model
@st.cache_resource
def load_sentiment_pipeline():
    model_name = "gsar78/HellenicSentimentAI"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_sentiment_pipeline()

# Fix common OCR misreads (Latin → Greek)
def correct_greek_misreads(text):
    substitutions = {
        "P": "Ρ", "H": "Η", "N": "Ν", "A": "Α", "B": "Β",
        "E": "Ε", "T": "Τ", "X": "Χ", "Y": "Υ", "M": "Μ"
    }
    for latin, greek in substitutions.items():
        text = text.replace(latin, greek)
    return text

# Google Vision OCR (document_text_detection, no hint)
def extract_text_google_vision(uploaded_file):
    uploaded_file.seek(0)
    content = uploaded_file.read()
    image = vision.Image(content=content)
    try:
        response = client.document_text_detection(image=image)
        return response.full_text_annotation.text if response.full_text_annotation.text else "No text found"
    except Exception as e:
        return f"❌ Vision API error: {e}"

# --- Streamlit Interface ---
st.title("🇬🇷 Greek Handwriting OCR + Sentiment Analysis")

uploaded_file = st.file_uploader("📄 Upload a handwritten Greek image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    uploaded_file.seek(0)
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)

    # OCR
    raw_text = extract_text_google_vision(uploaded_file)
    corrected_text = correct_greek_misreads(raw_text)

    # Display extracted text
    st.subheader("📝 Raw OCR Output")
    st.text_area("Raw Text", raw_text, height=150)

    st.subheader("🔠 Corrected Greek Text")
    st.text_area("Corrected Text", corrected_text, height=150)

    # Sentiment Analysis
    if corrected_text.strip() and "❌" not in corrected_text:
        sentiment = sentiment_pipeline(corrected_text[:512])[0]
        st.subheader("🧠 Sentiment Analysis")
        st.markdown(f"**Sentiment:** {sentiment['label']}  \n**Confidence:** {sentiment['score']:.2%}")
    else:
        st.info("No valid Greek text found for sentiment analysis.")
