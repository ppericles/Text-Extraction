# ==== utils_table.py ====

import pandas as pd
import numpy as np

def group_by_line(ocr_results, y_tolerance=12):
    """
    Groups OCR results into horizontal lines using Y overlap.

    Returns:
        List[List[OCR item]]
    """
    lines = []
    sorted_items = sorted(ocr_results, key=lambda x: x["bbox"][0][1])  # top Y

    for item in sorted_items:
        placed = False
        y = item["bbox"][0][1]
        for line in lines:
            if abs(line[0]["bbox"][0][1] - y) < y_tolerance:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])
    return lines

def group_by_columns(line, x_tolerance=30):
    """
    Groups a single OCR line into column cells based on X gaps.

    Returns:
        List[str] → cell text list
    """
    sorted_line = sorted(line, key=lambda x: x["bbox"][0][0])
    cells, current = [], ""
    prev_x = None

    for item in sorted_line:
        x = item["bbox"][0][0]
        if prev_x is not None and abs(x - prev_x) > x_tolerance:
            cells.append(current.strip())
            current = ""
        current += item["text"] + " "
        prev_x = x

    cells.append(current.strip())
    return cells

def extract_table_from_ocr(ocr_results, y_tolerance=12, x_tolerance=30):
    """
    Full pipeline to extract table rows (list of cell values) from OCR.

    Returns:
        List[List[str]] → table rows
    """
    lines = group_by_line(ocr_results, y_tolerance)
    rows = [group_by_columns(line, x_tolerance) for line in lines]
    return rows

def infer_table_headers(rows, min_columns=3):
    """
    Attempts to detect header row from OCR table rows.

    Returns:
        headers (List[str]), data_rows (List[List[str]])
    """
    for row in rows:
        if len(row) >= min_columns:
            headers = [cell.upper() for cell in row]
            return headers, rows[rows.index(row)+1:]
    return [], rows

def table_to_dataframe(headers, rows):
    """
    Converts header + rows into DataFrame.

    Returns:
        pd.DataFrame
    """
    return pd.DataFrame(rows, columns=headers[:len(rows[0])])

def export_table_json(headers, rows, filename="table.json"):
    """
    Saves table as structured JSON.

    Returns:
        None
    """
    table = [dict(zip(headers, row)) for row in rows]
    with open(filename, "w", encoding="utf-8") as f:
        import json
        json.dump(table, f, indent=2, ensure_ascii=False)
