# ==== FILE: utils_ocr.py ====

from google.cloud import vision
from google.cloud import documentai_v1beta3 as documentai
from PIL import Image
from io import BytesIO
import numpy as np

def parse_zone_text(image, engine="vision"):
    """
    Extracts raw text from a zone image using Google Vision OCR.
    """
    if engine == "vision":
        client = vision.ImageAnnotatorClient()
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        content = image_bytes.getvalue()
        image_obj = vision.Image(content=content)
        response = client.document_text_detection(image=image_obj)
        return response.full_text_annotation.text
    else:
        return "⚠️ Unknown OCR engine."

def form_parser_ocr(image: Image.Image, project_id: str, location: str, processor_id: str) -> dict:
    """
    Sends a PIL image to Google Document AI Form Parser and returns extracted key-value pairs with confidence scores.
    """
    client = documentai.DocumentProcessorServiceClient()
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")

    raw_document = documentai.RawDocument(content=image_bytes.getvalue(), mime_type="image/png")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)

    fields = {}
    for entity in result.document.entities:
        key = entity.type_.strip()
        value = entity.mention_text.strip()
        confidence = round(entity.confidence * 100)  # Convert to percentage
        fields[key] = {"value": value, "confidence": confidence}
    return fields

def match_fields_with_fallback(expected_keys, extracted_fields, image, layout_dict):
    """
    Matches expected field labels to extracted keys.
    If a field is missing, fallback to Vision OCR using layout box.
    """
    matched = {}
    for expected in expected_keys:
        for found_key in extracted_fields:
            if expected in found_key or found_key in expected:
                matched[expected] = extracted_fields[found_key]
                break
        else:
            # Fallback to Vision OCR
            box = layout_dict.get(expected)
            if box:
                w, h = image.size
                x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, [w, h, w, h])]
                cropped = image.crop((x1, y1, x2, y2))
                text = parse_zone_text(cropped, engine="vision").strip()
                matched[expected] = {"value": text, "confidence": 0}
            else:
                matched[expected] = {"value": "❌ Not found", "confidence": 0}
    return matched

def extract_fields_from_layout(image, layout_dict, engine="vision"):
    """
    Extracts field-level text from image using layout boxes.
    """
    client = vision.ImageAnnotatorClient()
    w, h = image.size
    fields = {}

    for label, box in layout_dict.items():
        x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, [w, h, w, h])]
        cropped = image.crop((x1, y1, x2, y2))
        image_bytes = BytesIO()
        cropped.save(image_bytes, format="PNG")
        content = image_bytes.getvalue()
        image_obj = vision.Image(content=content)
        response = client.document_text_detection(image=image_obj)
        text = response.full_text_annotation.text.strip()
        fields[label] = text
    return fields
