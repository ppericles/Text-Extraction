# =============================================================================
# FILE: utils_layout.py
# VERSION: 1.2
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Layout utilities for registry form parsing.
#              Includes layout validation, dummy box detection,
#              and visual overlays for debugging and production.
# =============================================================================

from PIL import Image, ImageDraw, ImageFont
from utils_ocr import vision_api_ocr, form_parser_ocr

# ----------------------------------------------------------------------
# 🔍 Field Extraction from Layout
# ----------------------------------------------------------------------

def extract_fields_from_layout(img: Image.Image, layout: dict, engine="vision", config=None) -> dict:
    """Extract text and confidence from layout boxes."""
    w, h = img.size
    results = {}

    for label, box in layout.items():
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

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

# ----------------------------------------------------------------------
# 🖼️ Standard Overlay Drawing
# ----------------------------------------------------------------------

def draw_layout_overlay(img: Image.Image, layout: dict) -> Image.Image:
    """Draw rectangles and labels over layout fields."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    for label, box in layout.items():
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            print(f"⚠️ Skipping invalid box for '{label}': {box}")
            continue

        x1, y1, x2, y2 = box
        left = int(x1 * w)
        top = int(y1 * h)
        right = int(x2 * w)
        bottom = int(y2 * h)

        draw.rectangle([left, top, right, bottom], outline="green", width=2)
        draw.text((left + 3, top + 3), label, fill="green")

    return overlay

# ----------------------------------------------------------------------
# ✅ Layout Validation for Preview Mode
# ----------------------------------------------------------------------

def validate_layout_for_preview(layout, image_width, image_height):
    """
    Ensures all zones in the layout contain a valid 4-point box.
    If missing or malformed, inserts a dummy box for preview purposes.
    """
    validated = {}
    default_box = [10, 10, 100, 40]  # Dummy box (x1, y1, x2, y2)

    for zone_name, zone_data in layout.items():
        box = zone_data.get("box") if isinstance(zone_data, dict) else zone_data

        if isinstance(box, (list, tuple)) and len(box) == 4:
            validated[zone_name] = {"box": box}
        else:
            print(f"⚠️ Zone '{zone_name}' missing or invalid box. Using dummy box.")
            offset = len(validated) * 50
            dummy_box = [
                10 + offset,
                10 + offset,
                100 + offset,
                40 + offset
            ]
            validated[zone_name] = {
                "box": dummy_box,
                "dummy": True
            }

    return validated

# ----------------------------------------------------------------------
# 🖍️ Preview Overlay with Dummy Box Detection
# ----------------------------------------------------------------------

def draw_layout_overlay_preview(image: Image.Image, layout: dict) -> Image.Image:
    """
    Draws bounding boxes over the image based on layout.
    Dummy boxes are shown in red dashed outlines with labels.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    w, h = image.size

    for zone_name, zone_data in layout.items():
        box = zone_data.get("box")
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(coord) for coord in box]
        is_dummy = zone_data.get("dummy", False) or (x2 - x1 <= 100 and y2 - y1 <= 50)

        if is_dummy:
            # Draw red dashed box
            dash_length = 5
            for i in range(x1, x2, dash_length * 2):
                draw.line([(i, y1), (i + dash_length, y1)], fill="red")
                draw.line([(i, y2), (i + dash_length, y2)], fill="red")
            for i in range(y1, y2, dash_length * 2):
                draw.line([(x1, i), (x1, i + dash_length)], fill="red")
                draw.line([(x2, i), (x2, i + dash_length)], fill="red")

            draw.text((x1 + 5, y1 + 5), f"{zone_name} (dummy)", fill="red", font=font)
        else:
            # Draw solid box with zone color
            color_map = {
                "group_a": "blue",
                "group_b": "green",
                "detail_zone": "orange",
                "master_zone": "purple"
            }
            color = color_map.get(zone_name, "gray")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 5, y1 + 5), zone_name, fill=color, font=font)

    return image
