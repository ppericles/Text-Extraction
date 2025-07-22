# ============================================================
# FILE: utils_image.py
# VERSION: 1.2
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Image preprocessing tools for registry parser.
#              Includes trimming, sharpening, resizing, and
#              zone splitting for master/detail layout.
# ============================================================

from PIL import Image, ImageOps, ImageEnhance
import numpy as np

# === Trim whitespace from edges ===
def trim_whitespace(img: Image.Image, threshold=240) -> Image.Image:
    gray = img.convert("L")
    np_img = np.array(gray)
    mask = np_img < threshold

    if not mask.any():
        return img

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped = img.crop((x0, y0, x1, y1))
    return cropped

# === Optimize image for OCR ===
def optimize_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")  # Grayscale
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    return img.convert("RGB")

# === Resize for Streamlit preview ===
def resize_for_preview(img: Image.Image, max_width=900) -> Image.Image:
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        return img.resize((int(w * ratio), int(h * ratio)))
    return img

# === Split image into master/detail zones ===
def split_zones_fixed(img: Image.Image, master_ratio=0.5):
    w, h = img.size
    master = img.crop((0, 0, w, int(h * master_ratio)))
    detail = img.crop((0, int(h * master_ratio), w, h))
    return (master, detail), {"master_height": int(h * master_ratio), "detail_height": h - int(h * master_ratio)}
