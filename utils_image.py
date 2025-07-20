# ==== utils_image.py ====

from PIL import Image, ImageOps
import numpy as np

# ✂️ Trim whitespace from image edges
def trim_whitespace(image, threshold=240):
    gray = image.convert("L")
    arr = np.array(gray)
    mask = arr < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image.crop((x0, y0, x1, y1))

# 🧩 Crop left half of image
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# 🧭 Split image into vertical zones with optional overlap
def split_zones_fixed(image, overlap=20):
    w, h = image.size
    zone_height = h // 3
    zones = []
    bounds = []

    for i in range(3):
        top = max(0, i * zone_height - overlap)
        bottom = min(h, (i + 1) * zone_height + overlap)
        zone = image.crop((0, top, w, bottom))
        zones.append(zone)
        bounds.append((top, bottom))

    return zones, bounds

# 🖼️ Resize image for preview
def resize_for_preview(image, max_width=800):
    w, h = image.size
    if w <= max_width:
        return image
    ratio = max_width / w
    return image.resize((int(w * ratio), int(h * ratio)))
