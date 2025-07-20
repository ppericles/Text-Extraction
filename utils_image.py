# ==== utils_image.py ====

from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2

# ✂️ Trim whitespace from edges
def trim_whitespace(image, threshold=240):
    gray = image.convert("L")
    arr = np.array(gray)
    mask = arr < threshold
    coords = np.argwhere(mask)
    if coords.size == 0: return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image.crop((x0, y0, x1, y1))

# 🧭 Vertical zone splitting (3 zones)
def split_zones_fixed(image, overlap=20):
    w, h = image.size
    zone_height = h // 3
    zones, bounds = [], []
    for i in range(3):
        top = max(0, i * zone_height - overlap)
        bottom = min(h, (i + 1) * zone_height + overlap)
        zones.append(image.crop((0, top, w, bottom)))
        bounds.append((top, bottom))
    return zones, bounds

# 🟨 Horizontal zone splitting (2 halves or columns)
def split_horizontal(image, split_ratio=0.5):
    w, h = image.size
    split_x = int(w * split_ratio)
    left = image.crop((0, 0, split_x, h))
    right = image.crop((split_x, 0, w, h))
    return left, right

# 🧼 Resize for compact preview
def resize_for_preview(image, max_width=800):
    w, h = image.size
    if w <= max_width: return image
    ratio = max_width / w
    return image.resize((int(w * ratio), int(h * ratio)))

# 🩹 Skew correction using OpenCV
def deskew_image(image, threshold=220):
    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] == 0: return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    rotated = Image.fromarray(cv2.warpAffine(
        np.array(image),
        cv2.getRotationMatrix2D((image.width // 2, image.height // 2), angle, 1.0),
        (image.width, image.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    ))
    return rotated

# 🖼️ Visual zone overlays (draw bounding boxes for inspection)
def draw_zones_overlays(image, bounds_list, box_color="blue"):
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for idx, (top, bottom) in enumerate(bounds_list, start=1):
        draw.rectangle([(0, top), (w, bottom)], outline=box_color, width=3)
        draw.text((10, top + 10), f"Zone {idx}", fill=box_color)
    return img
