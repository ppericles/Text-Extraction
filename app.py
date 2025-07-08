import streamlit as st
from PIL import Image
from transformers import pipeline
from google.cloud import vision
from google.oauth2 import service_account
import io

# Load Google Cloud credentials from Streamlit Secrets
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = vision.ImageAnnotatorClient(credentials=credentials)

# Function to extract text using Google Vision
def extract_text_google_vision(uploaded_file):
    uploaded_file.seek(0)  # Ensure pointer is at the beginning
    content = uploaded_file.read()

    if not content:
        return "⚠️ Uploaded file is empty."

    image = vision.Image(content=content)
    try:
        response = client.document_text_detection(image=image)
        return response.full_text_annotation.text if response.full_text_annotation.text else "No text found"
    except Exception as e:
        return f"❌ Vision API error: {e}"

# Load your Hugging Face model
classifier = pipeline("text-classification", model="ppericles/bert-template-classifier")

# Streamlit UI
st.title("📄 Greek Handwritten Template Classifier")

uploaded_file = st.file_uploader("Upload an image with handwritten Greek text", type=["png", "jpg", "jpeg"])

if uploaded_file:
    uploaded_file.seek(0)  # Reset file pointer before reading again
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # OCR
    text = extract_text_google_vision(uploaded_file)
    st.subheader("🧾 Extracted Text")
    st.text_area("Text", text, height=200)

    # Classification
    if text.strip():
        # Truncate text to 512 tokens using the tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("ppericles/bert-template-classifier")

        # Tokenize and truncate
        tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        truncated_text = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

        # Run classification
        prediction = classifier(truncated_text)[0]

        st.subheader("🧠 Predicted Template")
        st.write(f"**Label:** {prediction['label']}")
        st.write(f"**Confidence:** {prediction['score']:.2f}")
    else:
        st.warning("No text detected. Try a clearer image or different handwriting.")
