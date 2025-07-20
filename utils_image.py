# ==== FILE: utils_image.py - Image Cropping, Layout Tools, and Visual Overlays ====
# Version: 1.0.0
# Created: 2025-07-21
# Author: Pericles & Copilot
# Description: Functions for trimming whitespace, splitting zones, drawing overlays, and preparing image previews.

from PIL import Image, ImageDraw
import numpy as np

def resize_for_preview(image, max_width=600):
    w, h = image.size
    if w > max_width:
        ratio = max_width / w
        return image.resize((int(w * ratio), int(h * ratio)))
    return image

def trim_whitespace(image, threshold=240):
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
    Splits the full form image horizontally into Master and Detail zones.
    """
    w, h = image.size
    split_y = int(h * master_ratio)
    master_zone = image.crop((0, 0, w, split_y))
    detail_zone = image.crop((0, split_y, w, h))
    return [master_zone, detail_zone], [(0, 0, w, split_y), (0, split_y, w, h)]

def split_master_zone_vertically(image, split_ratio=0.3):
    """
    Splits the master zone vertically into Group A (left column) and Group B (right block).
    """
    w, h = image.size
    split_x = int(w * split_ratio)
    group_a = image.crop((0, 0, split_x, h))
    group_b = image.crop((split_x, 0, w, h))
    return group_a, group_b

def draw_colored_zones(image, master_bounds, detail_bounds, group_bounds=None):
    """
    Draws color-coded overlays for master/detail zones and optional groups A/B.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    draw.rectangle(master_bounds, outline="blue", width=3)
    draw.rectangle(detail_bounds, outline="gold", width=3)

    if group_bounds:
        draw.rectangle(group_bounds["group_a"], outline="brown", width=3)
        draw.rectangle(group_bounds["group_b"], outline="purple", width=3)

    return overlay

def draw_group_overlay(master_zone, split_ratio):
    """
    Draws overlay on master zone showing vertical split for Group A and Group B.
    """
    overlay = master_zone.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = master_zone.size
    split_x = int(w * split_ratio)
    draw.rectangle((0, 0, split_x, h), outline="brown", width=3)
    draw.rectangle((split_x, 0, w, h), outline="purple", width=3)
    return overlay
