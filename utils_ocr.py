# ==== utils_ocr.py ====

from io import BytesIO
from google.cloud import vision
from google.api_core.exceptions import ServiceUnavailable
from PIL import Image
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

    print(f"[OCR] Image type: {type(image)}")
    if not isinstance(image, Image.Image):
        return "❌ Invalid image object — must be a PIL.Image"

    try:
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        content = image_bytes.getvalue()
    except Exception as e:
        return f"❌ Failed to convert image to bytes: {str(e)}"

    for attempt in range(max_retries):
        try:
            response = client.document_text_detection(image=vision.Image(content=content))
            return response
        except ServiceUnavailable:
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
        return f"⚠️ Unsupported OCR engine: {engine}"

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

def extract_fields_from_layout(zone_img, layout_dict, engine="vision", **kwargs):
    """
    Extracts text from each field box defined in the layout.

    Args:
        zone_img (PIL.Image): Zone image
        layout_dict (dict): {label: [x1, y1, x2, y2]} in normalized coords
        engine (str): OCR engine name

    Returns:
        dict: {label: extracted_text}
    """
    assert isinstance(zone_img, Image.Image), "❌ zone_img must be a PIL.Image"

    w, h = zone_img.size
    results = {}

    for label, box in layout_dict.items():
        x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, [w, h, w, h])]
        field_crop = zone_img.crop((x1, y1, x2, y2))
        field_text = parse_zone_text(field_crop, engine=engine, **kwargs)
        results[label] = field_text.strip() if isinstance(field_text, str) else "⚠️ No text found"

    return results
