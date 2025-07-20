# ==== utils_export.py ====

import os
import pandas as pd

def export_forms_to_training_set(form_images, base_name, output_dir="training-set"):
    """
    Save cropped form images and manifest to disk.

    Args:
        form_images (list[PIL.Image]): Cropped form images
        base_name (str): Clean filename base (e.g. registry_001)
        output_dir (str): Destination folder
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    for idx, img in enumerate(form_images, start=1):
        form_id = f"{base_name}_form_{idx}"
        filename = f"{form_id}.png"
        path = os.path.join(output_dir, filename)
        img.save(path)
        manifest.append({
            "FormID": form_id,
            "Filename": filename
        })

    df = pd.DataFrame(manifest)
    df.to_csv(os.path.join(output_dir, "manifest.csv"), index=False)
