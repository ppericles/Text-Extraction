# =============================================================================
# FILE: utils_visualize.py
# VERSION: 1.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Visualization utilities for layout zones and clusters.
#              Draws overlays for grouped boxes and labeled zones.
# =============================================================================

from PIL import ImageDraw, ImageFont

def draw_clustered_boxes_overlay(image, clusters):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    palette = ["green", "blue", "red", "purple", "orange", "cyan"]

    for cluster_id, boxes in clusters.items():
        color = palette[cluster_id % len(palette)]
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 3, y1 + 3), f"C{cluster_id}-{i+1}", fill=color, font=font)

    return overlay

def draw_group_labels_overlay(image, layout):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    w, h = image.size
    color_map = {
        "group_a_box": "blue",
        "group_b_box": "red",
        "detail_top_box": "purple",
        "detail_bottom_box": "orange"
    }

    for label, color in color_map.items():
        box = layout.get(label)
        if box:
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, (w, h, w, h))]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 5, y1 + 5), label.replace("_box", "").upper(), fill=color, font=font)

    return overlay
