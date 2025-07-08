import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import os

from transformers import pipeline

# Load fine-tuned BERT model
classifier = pipeline("text-classification", model="./bert-template-classifier", tokenizer="./bert-template-classifier")

# Feedback CSV
FEEDBACK_FILE = "feedback_data.csv"
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["image_path", "text", "correct_template"]).to_csv(FEEDBACK_FILE, index=False)

# App UI
st.set_page_config(page_title="Intelligent OCR", layout="wide")
st.title("🧠 Intelligent OCR with Template Classification")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # OCR
    text = pytesseract.image_to_string(image)
    st.subheader("📄 Extracted Text")
    st.text_area("Text", text, height=200)

    # Template Prediction
    prediction = classifier(text[:512])[0]
    predicted_template = prediction["label"]
    st.subheader("🔍 Predicted Template")
    st.markdown(f"**{predicted_template}** (confidence: `{prediction['score']:.2f}`)")

    # Feedback
    st.subheader("✅ Confirm or Correct Template")
    correct_template = st.text_input("Correct Template", value=predicted_template)

    if st.button("Submit Feedback"):
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(f"{uploaded_file.name},{text.replace(',', ' ')},{correct_template}\n")
        st.success("✅ Feedback submitted! Thank you.")
