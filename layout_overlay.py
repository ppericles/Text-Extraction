# ==== layout_overlay.py ====

from PIL import ImageDraw, Image

def draw_layout_overlay(zone_img, layout_pixels, box_color="red", label_font=None):
    """
    Draws rectangles and labels over a zone image to preview field layouts.

    Args:
        zone_img (PIL.Image): The zone image.
        layout_pixels (dict): Dictionary of field labels to box tuples (x, y, w, h).
        box_color (str): Color for rectangle borders and text.
        label_font (ImageFont, optional): Font for drawing labels.

    Returns:
        PIL.Image: Annotated image.
    """
    img = zone_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for label, (x, y, w, h) in layout_pixels.items():
        rect = [(x, y), (x + w, y + h)]
        draw.rectangle(rect, outline=box_color, width=2)
        text_position = (x, max(0, y - 18))
        draw.text(text_position, label, fill=box_color, font=label_font)

    return img
