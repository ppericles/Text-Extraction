# ==== utils_mock.py ====

import os
from PIL import ImageDraw

def generate_mock_metadata_batch(layouts, expected_labels, count=1, placeholder="XXXX"):
    """
    Generates mock metadata rows based on layout templates.

    Args:
        layouts (dict): {zone_id: {label: [x1, y1, x2, y2]}}
        expected_labels (dict): {zone_id: [label1, label2, ...]}
        count (int): Number of mock rows to generate
        placeholder (str): Placeholder text

    Returns:
        list[dict]: List of mock metadata rows
    """
    rows = []
    for _ in range(count):
        row = {}
        for zid, labels in expected_labels.items():
            for label in labels:
                row[label] = placeholder
        rows.append(row)
    return rows

def export_mock_dataset_with_layout_overlay(
    metadata_rows,
    zones,
    layouts,
    ocr_traces,
    output_dir="training-set"
):
    """
    Exports zone images, overlays, and mock metadata to disk.

    Args:
        metadata_rows (list[dict]): Mock metadata
        zones (list[PIL.Image or None]): Zone images
        layouts (dict): Layouts per zone
        ocr_traces (dict): OCR results per form
        output_dir (str): Export folder
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in enumerate(metadata_rows):
        folder = os.path.join(output_dir, f"form_{idx + 1}")
        os.makedirs(folder, exist_ok=True)

        for zid, layout in layouts.items():
            zone_img = zones[int(zid) - 1]

            if zone_img is None:
                print(f"⚠️ Skipping zone {zid} — no image available.")
                continue

            # Save raw zone image
            zone_path = os.path.join(folder, f"zone_{zid}.png")
            zone_img.save(zone_path)

            # Draw overlay
            overlay = zone_img.copy()
            draw = ImageDraw.Draw(overlay)
            w, h = overlay.size

            for label, box in layout.items():
                x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, [w, h, w, h])]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1 + 5, y1 + 5), label, fill="red")

            overlay_path = os.path.join(folder, f"zone_{zid}_overlay.png")
            overlay.save(overlay_path)

        # Save OCR trace
        trace_path = os.path.join(folder, "ocr_trace.txt")
        with open(trace_path, "w", encoding="utf-8") as f:
            for line in ocr_traces.get(f"form_{idx + 1}", []):
                f.write(str(line) + "\n")

        # Save metadata
        meta_path = os.path.join(folder, "metadata.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            for key, value in row.items():
                f.write(f"{key}: {value}\n")
