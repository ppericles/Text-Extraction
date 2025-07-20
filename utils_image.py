# ==== utils_image.py ====

from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import cv2

def trim_whitespace(image, border=10):
    """
    Trim white margins from scanned image.
    """
    gray = image.convert("L")
    bw = np.array(gray) < 240
    coords = np.argwhere(bw)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = image.crop((x0 - border, y0 - border, x1 + border, y1 + border))
    return cropped

def deskew_image(image):
    """
    Deskew image using OpenCV moments.
    """
    img_cv = np.array(image.convert("L"))
    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle += 90
    center = tuple(np.array(image.size) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(np.array(image), rot_mat, image.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)

def split_zones_fixed(image, split_ratio=[0.3, 0.6]):
    """
    Split image into 3 vertical zones based on fixed ratios.
    """
    w, h = image.size
    y1 = int(h * split_ratio[0])
    y2 = int(h * split_ratio[1])
    zones = [
        image.crop((0, 0, w, y1)),
        image.crop((0, y1, w, y2)),
        image.crop((0, y2, w, h))
    ]
    bounds = [(0, y1), (y1, y2), (y2, h)]
    return zones, bounds

def draw_zones_overlays(image, bounds, color="green"):
    """
    Draw horizontal zone boundaries on image.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    w, _ = image.size
    for y1, y2 in bounds:
        draw.line([(0, y1), (w, y1)], fill=color, width=2)
        draw.line([(0, y2), (w, y2)], fill=color, width=2)
    return overlay

def draw_layout_overlay(image, layout, box_color="blue", text_color="white", font_size=14):
    """
    Draw labeled layout boxes on the image.

    Args:
        image (PIL.Image): Zone image
        layout (dict): Field → [x, y, w, h] normalized box
        box_color (str): Rectangle color
        text_color (str): Label text color
        font_size (int): Label font size

    Returns:
        PIL.Image: Image with overlay
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = image.size

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for label, box in layout.items():
        x, y, bw, bh = box
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + bw) * w), int((y + bh) * h)

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        draw.text((x1 + 4, y1 + 2), label, fill=text_color, font=font)

    return overlay

def resize_for_preview(image, max_width=800):
    """
    Resize image for Streamlit preview.
    """
    w, h = image.size
    if w > max_width:
        ratio = max_width / w
        return image.resize((int(w * ratio), int(h * ratio)))
    return image
