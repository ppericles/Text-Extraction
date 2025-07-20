# ==== FILE: utils_ocr.py - OCR Extraction and Field Matching Logic ====
# Version: 1.0.0
# Created: 2025-07-21
# Author: Pericles & Copilot
# Description: Handles Document AI and Vision OCR, and matches extracted text to expected fields with type-aware validation.

from google.cloud import documentai_v1 as documentai
from google.cloud import vision
from PIL import Image
import io
import re

def form_parser_ocr(image, project_id, location, processor_id):
    """
    Sends image to Document AI Form Parser and returns extracted fields.
    """
    client = documentai.DocumentUnderstandingServiceClient()
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    content = img_byte_arr.getvalue()

    raw_document = documentai.RawDocument(content=content, mime_type="image/png")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)

    fields = {}
    for entity in result.document.entities:
        fields[entity.type_] = {
            "value": entity.mention_text,
            "confidence": round(entity.confidence * 100, 2)
        }
    return fields

def vision_api_ocr(image):
    """
    Uses Google Vision API to extract raw text from an image.
    """
    client = vision.ImageAnnotatorClient()

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    content = img_byte_arr.getvalue()

    image_obj = vision.Image(content=content)
    response = client.text_detection(image=image_obj)

    if response.error.message:
        return ""

    return response.full_text_annotation.text.strip()

def is_valid(value, field_type):
    """
    Apply basic validation based on field type.
    """
    if not value:
        return False

    if field_type in ["Name", "Parent Name"]:
        return re.fullmatch(r"[Α-Ωα-ωΆΈΉΊΌΎΏΪΫ\- ]+", value)

    elif field_type == "ID":
        return re.fullmatch(r"[A-Z0-9\-]{4,}", value)

    elif field_type == "Date":
        return re.fullmatch(r"\d{2}/\d{2}/\d{4}", value) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)

    elif field_type == "Location":
        return not re.search(r"\d", value)

    return True  # Custom fields skip validation

def match_fields_with_fallback(expected_labels, extracted_fields, image, layout_dict):
    """
    Matches expected field labels to extracted values, using Vision OCR fallback and type-aware validation.
    """
    matched = {}
    for label in expected_labels:
        field_meta = layout_dict.get(label)
        box = field_meta.get("box") if field_meta else None
        field_type = field_meta.get("type", "Custom") if field_meta else "Custom"

        if label in extracted_fields:
            candidate = extracted_fields[label]["value"]
            confidence = extracted_fields[label]["confidence"]
            if is_valid(candidate, field_type):
                matched[label] = {"value": candidate, "confidence": confidence}
            else:
                matched[label] = {"value": candidate, "confidence": 0}
        elif box:
            w, h = image.size
            x1, y1, x2, y2 = box
            crop_box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
            cropped = image.crop(crop_box)
            fallback_text = vision_api_ocr(cropped).strip()
            if is_valid(fallback_text, field_type):
                matched[label] = {"value": fallback_text, "confidence": 0}
            else:
                matched[label] = {"value": fallback_text, "confidence": 0}
        else:
            matched[label] = {"value": "", "confidence": 0}
    return matched
