# ============================================================
# FILE: utils_layout.py
# VERSION: 1.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Layout visualization tools for registry parser.
#              Includes bounding box overlays, column break
#              visualization, and label rendering.
# ============================================================
from PIL import ImageDraw, ImageFont

# === Draw bounding boxes on image ===
def draw_boxes(img, boxes, color="red", width=2):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle(
            [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)],
            outline=color,
            width=width
        )
    return img

# === Draw vertical column breaks ===
def draw_column_breaks(img, column_breaks, color="blue", width=2):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for x1, _ in column_breaks:
        x = int(x1 * w)
        draw.line([(x, 0), (x, h)], fill=color, width=width)
    return img

# === Overlay labels and confidence scores ===
def draw_labels(img, fields, box, font_size=14):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x1, y1, x2, y2 = box
    zone_w = int((x2 - x1) * w)
    zone_h = int((y2 - y1) * h)
    zone_x = int(x1 * w)
    zone_y = int(y1 * h)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    y_offset = zone_y + 5
    for label, data in fields.items():
        text = f"{label}: {data['value']} ({data['confidence']}%)"
        draw.text((zone_x + 5, y_offset), text, fill="black", font=font)
        y_offset += font_size + 4

    return img
