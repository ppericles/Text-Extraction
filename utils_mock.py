# ==== utils_mock.py ====

import os, json
from PIL import Image
from utils_image import draw_layout_overlay
from utils_layout import LayoutManager

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

def export_mock_dataset_with_layout_overlay(rows, zones, box_layouts, ocr_traces, output_dir="training-set"):
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

        # Save zone images + overlays
        for i, zone_img in enumerate(zones, start=1):
            zid = str(i)
            zone_img.save(os.path.join(folder, f"zone_{zid}.png"))

            layout = box_layouts.get(zid, {})
            if layout:
                manager = LayoutManager(zone_img.size)
                overlay = draw_layout_overlay(zone_img, manager.load_layout(layout))
                overlay.save(os.path.join(folder, f"zone_{zid}_overlay.png"))
