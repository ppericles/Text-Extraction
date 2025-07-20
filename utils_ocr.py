# ==== utils_ocr.py ====

from io import BytesIO
from google.cloud import vision
from google.api_core.exceptions import ServiceUnavailable
import time

def run_vision_ocr(image, max_retries=3):
    """
    Runs OCR using Google Cloud Vision API with retry logic.

    Args:
        image (PIL.Image): Zone image
        max_retries (int): Number of retry attempts on failure

    Returns:
        vision.AnnotateImageResponse or str: OCR response or error message
    """
    client = vision.ImageAnnotatorClient()

    # Convert PIL image to bytes
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    content = image_bytes.getvalue()

    for attempt in range(max_retries):
        try:
            response = client.document_text_detection(image=vision.Image(content=content))
            return response
        except ServiceUnavailable as e:
            time.sleep(2 ** attempt)
        except Exception as e:
            return f"❌ OCR error: {str(e)}"

    return "❌ OCR failed after multiple retries."

def run_ocr(image, engine="vision", **kwargs):
    """
    Dispatches OCR to the selected engine.

    Args:
        image (PIL.Image): Zone image
        engine (str): OCR engine name

    Returns:
        OCR result or error message
    """
    if engine == "vision":
        return run_vision_ocr(image, **kwargs)
    else:
        return "⚠️ Unsupported OCR engine."

def parse_zone_text(image, engine="vision", **kwargs):
    """
    Parses text from a zone image using OCR.

    Args:
        image (PIL.Image): Zone image
        engine (str): OCR engine name

    Returns:
        str: Extracted text or error message
    """
    ocr_results = run_ocr(image, engine=engine, **kwargs)

    if isinstance(ocr_results, str):
        return ocr_results  # Error message

    try:
        return ocr_results.full_text_annotation.text
    except AttributeError:
        return "⚠️ No text found in OCR response."
