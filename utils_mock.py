# ==== utils_mock.py ====

import os, json
from PIL import Image

def generate_mock_metadata_row(box_layouts, expected_labels, placeholder="XXXX", form_id="mock_001"):
    row = {}
    for zid, labels in expected_labels.items():
        for label in labels:
            row[label] = placeholder if label in box_layouts.get(zid, {}) else "---"
    row["FormID"] = form_id
    return row

def generate_mock_metadata_batch(box_layouts, expected_labels, count=10, placeholder="XXXX"):
    rows = []
    for i in range(1, count + 1):
        form_id = f"mock_{i:03d}"
        row = generate_mock_metadata_row(box_layouts, expected_labels, placeholder, form_id)
        rows.append(row)
    return rows

def export_mock_dataset(rows, zones, output_dir="training-set"):
    """
    Save each mock row as JSON + blank zone images.

    Args:
        rows (list): List of metadata dicts
        zones (list): List of PIL.Image zones [zone1, zone2, zone3]
        output_dir (str): Root folder to export dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    for row in rows:
        form_id = row["FormID"]
        folder = os.path.join(output_dir, form_id)
        os.makedirs(folder, exist_ok=True)

        # Save metadata
        with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)

        # Save zone images
        for i, zone_img in enumerate(zones, start=1):
            zone_path = os.path.join(folder, f"zone_{i}.png")
            zone_img.save(zone_path)

def export_mock_dataset_with_ocr(rows, zones, ocr_traces, output_dir="training-set"):
    """
    Save each mock row + OCR trace + zone images.

    Args:
        rows (list): List of metadata dicts
        zones (list): List of PIL.Image zones [zone1, zone2, zone3]
        ocr_traces (dict): FormID → list of OCR dicts per zone
        output_dir (str): Root folder to export dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    for row in rows:
        form_id = row["FormID"]
        folder = os.path.join(output_dir, form_id)
        os.makedirs(folder, exist_ok=True)

        # Save metadata
        with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)

        # Save OCR trace
        trace = ocr_traces.get(form_id, [])
        with open(os.path.join(folder, "ocr_trace.json"), "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

        # Save zone images
        for i, zone_img in enumerate(zones, start=1):
            zone_img.save(os.path.join(folder, f"zone_{i}.png"))
