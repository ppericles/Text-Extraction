import streamlit as st
from PIL import Image
from transformers import pipeline
from google.cloud import vision
from google.oauth2 import service_account
import io

def correct_greek_misreads(text):
    substitutions = {
        "P": "Ρ",  # Latin P → Greek Rho
        "H": "Η",  # Latin H → Greek Eta
        "N": "Ν",  # Latin N → Greek Nu
        "A": "Α",  # Latin A → Greek Alpha
        "B": "Β",  # Latin B → Greek Beta
        "E": "Ε",  # Latin E → Greek Epsilon
        "T": "Τ",  # Latin T → Greek Tau
        "X": "Χ",  # Latin X → Greek Chi
        "Y": "Υ",  # Latin Y → Greek Upsilon
        "M": "Μ",  # Latin M → Greek Mu
    }
    for latin, greek in substitutions.items():
        text = text.replace(latin, greek)
    return text

from PIL import ImageEnhance, ImageOps

def preprocess_image(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("L")  # Grayscale
    image = ImageOps.autocontrast(image)            # Enhance contrast
    image = image.resize((1024, int(1024 * image.height / image.width)))  # Resize
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

# Load Google Cloud credentials from Streamlit Secrets
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = vision.ImageAnnotatorClient(credentials=credentials)

# Function to extract text using Google Vision
def extract_text_google_vision(uploaded_file):
    uploaded_file.seek(0)
    content = preprocess_image(uploaded_file)
    if not content:
        return "⚠️ Uploaded file is empty."

    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["el"])  # Greek language hint

    try:
        response = client.text_detection(image=image)
        return response.text_annotations[0].description if response.text_annotations else "No text found"
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
    raw_text = extract_text_google_vision(uploaded_file)
    text = correct_greek_misreads(raw_text)
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
