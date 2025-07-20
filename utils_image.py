# ==== FILE: utils_image.py ====

from PIL import Image, ImageDraw
import numpy as np

def resize_for_preview(image, max_width=600):
    """
    Resize image for display in Streamlit preview.
    """
    w, h = image.size
    if w > max_width:
        ratio = max_width / w
        return image.resize((int(w * ratio), int(h * ratio)))
    return image

def trim_whitespace(image, threshold=240):
    """
    Trims white borders from scanned form image.
    """
    gray = image.convert("L")
    np_img = np.array(gray)
    mask = np_img < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image.crop((x0, y0, x1, y1))

def split_zones_fixed(image, master_ratio=0.5):
    """
    Splits form image into master and detail zones vertically.
    """
    w, h = image.size
    split_y = int(h * master_ratio)
    master_zone = image.crop((0, 0, w, split_y))
    detail_zone = image.crop((0, split_y, w, h))
    return [master_zone, detail_zone], [(0, 0, w, split_y), (0, split_y, w, h)]

def split_master_zone_vertically(image, split_ratio=0.3):
    """
    Splits master zone vertically into Group A (left) and Group B (right).
    """
    w, h = image.size
    split_x = int(w * split_ratio)
    group_a = image.crop((0, 0, split_x, h))         # Left column
    group_b = image.crop((split_x, 0, w, h))          # Right column
    return group_a, group_b

def draw_zones_overlays(image, bounds, color="red"):
    """
    Draws bounding boxes over zones for visual preview.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for box in bounds:
        draw.rectangle(box, outline=color, width=3)
    return overlay
