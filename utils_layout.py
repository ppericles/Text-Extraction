# ============================================================
# FILE: utils_layout.py
# VERSION: 1.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Layout utilities for registry form parsing.
#              Supports Vision API and Document AI OCR,
#              confidence scoring, and visual overlays.
# ============================================================

from PIL import Image, ImageDraw
from utils_ocr import vision_api_ocr, form_parser_ocr

def extract_fields_from_layout(img: Image.Image, layout: dict, engine="vision", config=None) -> dict:
    """Extract text and confidence from layout boxes."""
    w, h = img.size
    results = {}

    for label, box in layout.items():
        x1, y1, x2, y2 = box
        left = int(x1 * w)
        top = int(y1 * h)
        right = int(x2 * w)
        bottom = int(y2 * h)

        crop = img.crop((left, top, right, bottom))

        if engine == "documentai" and config:
            fields = form_parser_ocr(crop, **config)
            best = max(fields, key=lambda f: f.get("confidence", 0), default={})
            results[label] = {
                "value": best.get("value", ""),
                "confidence": best.get("confidence", 0)
            }
        else:
            text = vision_api_ocr(crop).strip()
            results[label] = {
                "value": text,
                "confidence": 100  # Vision API doesn't return confidence
            }

    return results

def draw_layout_overlay(img: Image.Image, layout: dict) -> Image.Image:
    """Draw rectangles and labels over layout fields."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    for label, box in layout.items():
        x1, y1, x2, y2 = box
        left = int(x1 * w)
        top = int(y1 * h)
        right = int(x2 * w)
        bottom = int(y2 * h)

        draw.rectangle([left, top, right, bottom], outline="green", width=2)
        draw.text((left + 3, top + 3), label, fill="green")

    return overlay
