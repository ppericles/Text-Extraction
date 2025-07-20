# ==== FILE: utils_image.py ====

from PIL import Image, ImageDraw, ImageChops

def trim_whitespace(image):
    """
    Trims white borders from an image using background subtraction.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("❌ Input must be a PIL.Image")

    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    return image.crop(bbox) if bbox else image

def split_zones_fixed(image, master_ratio=0.5):
    """
    Splits image vertically into master/detail zones.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("❌ Input must be a PIL.Image")

    w, h = image.size
    if w == 0 or h == 0:
        raise ValueError("❌ Image has invalid dimensions")

    split_y = int(h * master_ratio)
    zone1 = image.crop((0, 0, w, split_y))
    zone2 = image.crop((0, split_y, w, h))
    zone3 = None  # Optional third zone
    return [zone1, zone2, zone3], [(0, 0, w, split_y), (0, split_y, w, h)]

def draw_zones_overlays(image, bounds):
    """
    Draws colored overlays for each zone.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    colors = ["blue", "green", "orange"]
    for i, box in enumerate(bounds):
        draw.rectangle(box, outline=colors[i % len(colors)], width=3)
        draw.text((box[0] + 5, box[1] + 5), f"Zone {i+1}", fill=colors[i % len(colors)])
    return overlay

def draw_layout_overlay(image, layout_dict):
    """
    Draws layout boxes on the image.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    for label, box in layout_dict.items():
        x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, [w, h, w, h])]
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1 + 5, y1 + 5), label, fill="green")
    return overlay

def draw_invalid_boxes_overlay(image, layout_dict):
    """
    Draws red boxes for invalid layout regions on the image.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    for label, box in layout_dict.items():
        x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, [w, h, w, h])]
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 5, y1 + 5), f"❌ {label}", fill="red")
    return overlay

def resize_for_preview(image, max_width=800):
    """
    Resizes image for display in Streamlit.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("❌ Input must be a PIL.Image")

    w, h = image.size
    if w == 0 or h == 0:
        raise ValueError("❌ Image has invalid dimensions")

    if w > max_width:
        ratio = max_width / w
        return image.resize((int(w * ratio), int(h * ratio)))
    return image
