# FILE: utils_ocr.py
# Description: OCR utilities with improved error handling
# Version: 1.1

import io
from PIL import Image
from google.cloud import documentai_v1 as documentai
from google.api_core.exceptions import GoogleAPIError

def image_to_bytes(img: Image.Image) -> bytes:
    # Convert image to bytes (force RGB for compatibility)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def form_parser_ocr(img: Image.Image, project_id: str, location: str, processor_id: str) -> dict:
    try:
        if not all([project_id, location, processor_id]):
            raise ValueError("Missing Document AI configuration (project_id, location, processor_id)")

        client = documentai.DocumentProcessorServiceClient()
        name = client.processor_path(project_id, location, processor_id)

        image_bytes = image_to_bytes(img)

        raw_document = documentai.RawDocument(
            content=image_bytes,
            mime_type="image/png"
        )

        request = documentai.ProcessRequest(
            name=name,
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

        return fields

    except GoogleAPIError as api_error:
        print(f"[Document AI] API error: {api_error}")
        return {}

    except Exception as e:
        print(f"[Document AI] Processing failed: {e}")
        return {}
