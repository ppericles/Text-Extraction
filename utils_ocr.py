# ============================================================
# FILE: utils_ocr.py
# VERSION: 1.3
# AUTHOR: Pericles & Copilot
# DESCRIPTION: OCR utilities for registry parser. Includes
#              Document AI integration, Vision API fallback,
#              field matching, confidence scoring, and box extraction.
# ============================================================

import io
from PIL import Image
from google.cloud import documentai_v1 as documentai
from google.cloud import vision
from google.api_core.exceptions import GoogleAPIError

# === Helper: Convert image to bytes ===
def image_to_bytes(img: Image.Image) -> bytes:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

# === Primary OCR via Document AI ===
def form_parser_ocr(img: Image.Image, project_id: str, location: str, processor_id: str) -> dict:
    try:
        if not all([project_id, location, processor_id]):
            raise ValueError("Missing Document AI configuration.")

        client = documentai.DocumentProcessorServiceClient()
        processor_path = client.processor_path(project_id, location, processor_id)

        image_bytes = image_to_bytes(img)

        raw_document = documentai.RawDocument(
            content=image_bytes,
            mime_type="image/png"
        )

        request = documentai.ProcessRequest(
            name=processor_path,
            raw_document=raw_document
        )

        result = client.process_document(request=request)
        document = result.document

        fields = {}
        for entity in document.entities:
            label = entity.type_
            value = entity.mention_text or ""
            score = round(entity.confidence * 100)
            if label not in fields or score > fields[label]["confidence"]:
                fields[label] = {"value": value, "confidence": score}

        return document.to_dict()  # 🔄 Return full document for layout parsing

    except GoogleAPIError as api_error:
        print(f"[Document AI] API error: {api_error}")
        return {}

    except Exception as e:
        print(f"[Document AI] Unexpected error: {e}")
        return {}

# === Fallback OCR via Vision API ===
def vision_api_ocr(img: Image.Image) -> str:
    try:
        client = vision.ImageAnnotatorClient()
        image_bytes = image_to_bytes(img)
        image = vision.Image(content=image_bytes)

        response = client.text_detection(image=image)
        annotations = response.text_annotations

        if annotations:
            return annotations[0].description.strip()
        return ""

    except Exception as e:
        print(f"[Vision API] OCR failed: {e}")
        return ""

# === Match expected fields with OCR output ===
def match_fields_with_fallback(expected_fields, parsed_fields, img, layout) -> dict:
    matched = {}
    for key in expected_fields:
        if key in parsed_fields:
            matched[key] = parsed_fields[key]
        else:
            matched[key] = {"value": "", "confidence": 0}
    return matched

# === Extract bounding boxes from Document AI ===
def documentai_ocr_boxes(response: dict, image: Image.Image, page_number=1) -> tuple:
    boxes = []
    w, h = image.size

    for page in response.get("pages", []):
        if page.get("pageNumber", 1) != page_number:
            continue

        for element in page.get("lines", []):
            layout = element.get("layout", {})
            poly = layout.get("boundingPoly", {}).get("normalizedVertices", [])

            if len(poly) == 4:
                x_coords = [int(v.get("x", 0) * w) for v in poly]
                y_coords = [int(v.get("y", 0) * h) for v in poly]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                boxes.append((x1, y1, x2, y2))

    return boxes, None

# === Extract bounding boxes from Vision API ===
def vision_api_ocr_boxes(img: Image.Image) -> list:
    try:
        client = vision.ImageAnnotatorClient()
        image_bytes = image_to_bytes(img)
        image = vision.Image(content=image_bytes)

        response = client.document_text_detection(image=image)
        boxes = []

        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        vertices = word.bounding_box.vertices
                        if len(vertices) == 4:
                            x_coords = [v.x for v in vertices]
                            y_coords = [v.y for v in vertices]
                            x1, x2 = min(x_coords), max(x_coords)
                            y1, y2 = min(y_coords), max(y_coords)
                            boxes.append((x1, y1, x2, y2))
        return boxes

    except Exception as e:
        print(f"[Vision API] Box extraction failed: {e}")
        return []
