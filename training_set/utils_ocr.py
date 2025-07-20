# ==== utils_ocr.py ====

import io
from PIL import Image
from google.cloud import documentai_v1 as docai
from google.cloud import vision

# 🧠 OCR Engine Selector
def run_ocr(image, engine="documentai", project_id=None, processor_id=None, location="us"):
    if engine == "documentai":
        return run_documentai_ocr(image, project_id, processor_id, location)
    elif engine == "vision":
        return run_vision_ocr(image)
    else:
        raise ValueError("Unsupported OCR engine")

# 📄 Google Document AI OCR
def run_documentai_ocr(image, project_id, processor_id, location="us"):
    client = docai.DocumentProcessorServiceClient()
    name = client.processor_path(project_id, location, processor_id)

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    raw_document = docai.RawDocument(content=image_bytes.getvalue(), mime_type="image/png")
    request = docai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)

    fields = []
    for entity in result.document.entities:
        fields.append({
            "text": entity.mention_text,
            "type": entity.type_,
            "confidence": entity.confidence,
            "bounding_box": extract_bbox(entity.page_anchor)
        })
    return fields

# 📷 Google Vision OCR
def run_vision_ocr(image):
    client = vision.ImageAnnotatorClient()
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    content = image_bytes.getvalue()

    response = client.document_text_detection(image=vision.Image(content=content))
    fields = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = "".join([symbol.text for symbol in word.symbols])
                    confidence = word.confidence
                    bbox = [(v.x, v.y) for v in word.bounding_box.vertices]
                    fields.append({
                        "text": text,
                        "confidence": confidence,
                        "bounding_box": bbox
                    })
    return fields

# 📐 Bounding Box Extractor for Document AI
def extract_bbox(page_anchor):
    if not page_anchor.page_refs: return None
    ref = page_anchor.page_refs[0]
    if not ref.bounding_poly: return None
    return [(v.x, v.y) for v in ref.bounding_poly.vertices]

# 🧭 Zone-Level Parser
def parse_zone_text(zone_img, engine="documentai", **kwargs):
    ocr_results = run_ocr(zone_img, engine=engine, **kwargs)
    return [
        {
            "text": item["text"],
            "confidence": item.get("confidence", 0.0),
            "bbox": item.get("bounding_box", [])
        }
        for item in ocr_results
        if item.get("text")
    ]
