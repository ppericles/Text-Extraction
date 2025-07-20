# ==== utils_image.py ====

from PIL import Image, ImageDraw

def trim_whitespace(image, threshold=240):
    """
    Trims white borders from the image.
    """
    assert isinstance(image, Image.Image), "❌ trim_whitespace expects a PIL.Image"
    gray = image.convert("L")
    bbox = gray.point(lambda x: x < threshold and 255).getbbox()
    return image.crop(bbox) if bbox else image

def split_zones_fixed(image, master_ratio=0.5):
    """
    Splits image into two vertical zones: master and detail.

    Args:
        image (PIL.Image): The cropped form image
        master_ratio (float): Ratio of vertical space for master zone (0.3–0.7)

    Returns:
        zones: [zone1, zone2, zone3]
        bounds: [top_y, split_y, bottom_y]
    """
    assert isinstance(image, Image.Image), "❌ split_zones_fixed expects a PIL.Image"

    h = image.height
    split_y = int(h * master_ratio)
    zone1 = image.crop((0, 0, image.width, split_y))
    zone2 = image.crop((0, split_y, image.width, h))
    zone3 = None  # Reserved

    bounds = [0, split_y, h]
    return [zone1, zone2, zone3], bounds

def draw_zones_overlays(image, bounds):
    """
    Draws green horizontal lines to indicate zone boundaries.

    Args:
        image (PIL.Image): The full form image
        bounds (list[int]): Y-coordinates of zone splits

    Returns:
        PIL.Image: Image with overlay lines
    """
    assert isinstance(image, Image.Image), "❌ draw_zones_overlays expects a PIL.Image"

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for y in bounds[1:-1]:  # Skip top and bottom
        draw.line([(0, y), (image.width, y)], fill="green", width=3)

    return overlay

def draw_layout_overlay(image, layout_dict):
    """
    Draws red boxes and labels over a zone image using normalized coordinates.

    Args:
        image (PIL.Image): Zone image
        layout_dict (dict): {label: [x1, y1, x2, y2]} in normalized coords

    Returns:
        PIL.Image: Image with layout overlay
    """
    assert isinstance(image, Image.Image), "❌ draw_layout_overlay expects a PIL.Image"

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = image.size

    for label, box in layout_dict.items():
        x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, [w, h, w, h])]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 5, y1 + 5), label, fill="red")

    return overlay

def resize_for_preview(image, max_width=800):
    """
    Resizes image for Streamlit preview without distortion.

    Args:
        image (PIL.Image): Original image
        max_width (int): Max width for display

    Returns:
        PIL.Image: Resized image
    """
    assert isinstance(image, Image.Image), "❌ resize_for_preview expects a PIL.Image"

    w, h = image.size
    if w <= max_width:
        return image
    ratio = max_width / w
    return image.resize((int(w * ratio), int(h * ratio)))
