# ============================================================
# FILE: utils_parser.py
# VERSION: 1.1
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Form-level parsing logic for registry scans.
#              Crops using absolute pixel coordinates.
# ============================================================

from utils_ocr import form_parser_ocr, match_fields_with_fallback, vision_api_ocr
from utils_image import split_zones_fixed
import cv2
import numpy as np

def suggest_column_breaks(pil_image, threshold=220, min_gap=15):
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    profile = np.sum(binary, axis=0)
    gaps = np.where(profile < np.max(profile) * 0.1)[0]
    separators, prev = [], None
    for idx in gaps:
        if prev is None or idx - prev > min_gap:
            separators.append(idx)
        prev = idx
    columns = []
    for i in range(len(separators) - 1):
        x1, x2 = separators[i] / binary.shape[1], separators[i + 1] / binary.shape[1]
        if x2 - x1 > 0.05:
            columns.append((x1, x2))
    return columns

def extract_table(table_img, config, column_breaks, rows=10, header=True):
    w, h = table_img.size
    row_h = h / (rows + int(header))
    headers, data = [], []
    for r in range(rows + int(header)):
        y1, y2 = int(r * row_h), int((r + 1) * row_h)
        row = {}
        for i, (x1, x2) in enumerate(column_breaks):
            cell = table_img.crop((int(x1 * w), y1, int(x2 * w), y2))
            text = vision_api_ocr(cell).strip()
            if header and r == 0:
                headers.append(text or f"col_{i}")
            elif r >= int(header):
                key = headers[i] if i < len(headers) else f"col_{i}"
                row[key] = text
        if r >= int(header):
            data.append(row)
    return headers, data

def process_single_form(image, box, index, config, layout):
    # box contains absolute pixel coordinates
    x1, y1, x2, y2 = box
    form_crop = image.crop((x1, y1, x2, y2))
    zones, _ = split_zones_fixed(form_crop, layout.get("master_ratio", 0.5))
    master_zone, detail_zone = zones

    gw, gh = master_zone.size
    xa1, ya1, xa2, ya2 = layout.get("group_a_box", [0.0, 0.0, 0.2, 1.0])
    xb1, yb1, xb2, yb2 = layout.get("group_b_box", [0.2, 0.0, 1.0, 0.5])
    group_a = master_zone.crop((int(xa1*gw), int(ya1*gh), int(xa2*gw), int(ya2*gh)))
    group_b = master_zone.crop((int(xb1*gw), int(yb1*gh), int(xb2*gw), int(yb2*gh)))

    dw, dh = detail_zone.size
    xt1, yt1, xt2, yt2 = layout.get("detail_box", [0.0, 0.0, 1.0, 1.0])
    detail_crop = detail_zone.crop((int(xt1*dw), int(yt1*dh), int(xt2*dw), int(yt2*dh)))

    expected_a = ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"]
    expected_b = ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]
    fields_a = form_parser_ocr(group_a, **config)
    fields_b = form_parser_ocr(group_b, **config)
    matched_a = match_fields_with_fallback(expected_a, fields_a, group_a, layout)
    matched_b = match_fields_with_fallback(expected_b, fields_b, group_b, layout)

    meridos = matched_a.get("ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", {}).get("value", "")
    if not meridos and index > 0:
        prev = layout.get(f"form_{index-1}_meridos", "")
        if prev.isdigit():
            meridos = str(int(prev) + 1)
            matched_a["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"] = {"value": meridos, "confidence": 0}
    layout[f"form_{index}_meridos"] = meridos

    column_breaks = layout.get("table_columns")
    if layout.get("auto_detect", True):
        column_breaks = suggest_column_breaks(detail_crop)

    _, table_rows = extract_table(detail_crop, config, column_breaks)

    return {
        "master": master_zone,
        "detail": detail_zone,
        "group_a": matched_a,
        "group_b": matched_b,
        "table_crop": detail_crop,
        "column_breaks": column_breaks,
        "table_rows": table_rows
    }
