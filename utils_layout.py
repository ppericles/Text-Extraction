# =============================================================================
# FILE: utils_layout.py
# VERSION: 2.1
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Layout utilities for registry form parsing.
#              Includes auto layout detection, bounding box extraction,
#              and overlay visualization for Vision API and Document AI.
# =============================================================================

from PIL import Image, ImageDraw, ImageFont
from utils_ocr import vision_api_ocr_boxes, form_parser_ocr, documentai_ocr_boxes

# === Auto Layout Detection ===
def auto_detect_layout(img: Image.Image, use_docai=False, config=None) -> dict:
    w, h = img.size

    if use_docai and config:
        response = form_parser_ocr(img, **config)
        boxes, _ = documentai_ocr_boxes(response, img)
    else:
        boxes = vision_api_ocr_boxes(img)

    if not boxes:
        return {
            "master_box": [0.0, 0.0, 1.0, 0.5],
            "detail_box": [0.0, 0.5, 1.0, 1.0]
        }

    centers = [(y1 + y2) / 2 / h for _, y1, _, y2 in boxes]
    centers.sort()
    gaps = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
    max_gap_idx = gaps.index(max(gaps)) if gaps else 0
    split_y = (centers[max_gap_idx] + centers[max_gap_idx + 1]) / 2 if gaps else 0.5

    return {
        "master_box": [0.0, 0.0, 1.0, split_y],
        "detail_box": [0.0, split_y, 1.0, 1.0]
    }

# === Overlay Drawing ===
def draw_layout_overlay(image: Image.Image, layout: dict) -> Image.Image:
    """
    Draws layout zones on the image using colored rectangles.
    Supports master/detail and optional subzones.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    w, h = image.size

    color_map = {
        "master_box": "green",
        "detail_box": "gray",
        "group_a_box": "blue",
        "group_b_box": "red",
        "detail_top_box": "purple",
        "detail_bottom_box": "orange"
    }

    for label, color in color_map.items():
        box = layout.get(label)
        if box and len(box) == 4:
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, (w, h, w, h))]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 5, y1 + 5), label.replace("_box", "").upper(), fill=color, font=font)

    return overlay
