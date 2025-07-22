# ============================================================
# FILE: utils_image.py
# VERSION: 1.4
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Image preprocessing tools for registry parser.
#              Includes trimming, resizing, layout splitting,
#              column/row break visualization, and adaptive trim.
# ============================================================

from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import numpy as np

# === Trim whitespace from edges (fixed threshold) ===
def trim_whitespace(img: Image.Image, threshold=240) -> Image.Image:
    gray = img.convert("L")
    np_img = np.array(gray)
    mask = np_img < threshold

    if not mask.any():
        return img

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img.crop((x0, y0, x1, y1))

# === Adaptive trim based on content density ===
def adaptive_trim_whitespace(img: Image.Image, density_threshold=0.02) -> Image.Image:
    gray = img.convert("L")
    np_img = np.array(gray)
    binary = np_img < 200
    vertical_density = np.mean(binary, axis=1)
    horizontal_density = np.mean(binary, axis=0)

    y_indices = np.where(vertical_density > density_threshold)[0]
    x_indices = np.where(horizontal_density > density_threshold)[0]

    if len(y_indices) == 0 or len(x_indices) == 0:
        return img

    y0, y1 = y_indices[0], y_indices[-1] + 1
    x0, x1 = x_indices[0], x_indices[-1] + 1

    return img.crop((x0, y0, x1, y1))

# === Optimize image for OCR ===
def optimize_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
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
    return (master, detail), {
        "master_height": int(h * master_ratio),
        "detail_height": h - int(h * master_ratio)
    }

# === Draw column breaks on detail zone ===
def draw_column_breaks(img: Image.Image, column_breaks: list, color="blue") -> Image.Image:
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    for x1, x2 in column_breaks:
        draw.line([(int(x1 * w), 0), (int(x1 * w), h)], fill=color, width=2)
        draw.line([(int(x2 * w), 0), (int(x2 * w), h)], fill=color, width=2)
    return overlay

# === Draw row breaks on detail zone ===
def draw_row_breaks(img: Image.Image, rows=10, header=True, color="red") -> Image.Image:
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    total_rows = rows + int(header)
    row_height = h / total_rows
    for r in range(total_rows + 1):
        y = int(r * row_height)
        draw.line([(0, y), (w, y)], fill=color, width=2)
    return overlay
