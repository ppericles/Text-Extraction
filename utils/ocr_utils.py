from unidecode import unidecode
import numpy as np

def normalize(text):
    return unidecode(text.upper().strip())

def detect_header_regions(annotations, field_labels, layout_dict):
    normalized_labels = {normalize(lbl): lbl for lbl in field_labels}
    for ann in annotations[1:]:  # skip full text block
        txt = normalize(ann.description)
        if txt in normalized_labels:
            label = normalized_labels[txt]
            vertices = ann.bounding_poly.vertices
            xs = [int(v.x) for v in vertices if v.x is not None]
            ys = [int(v.y) for v in vertices if v.y is not None]
            if len(xs) == 4 and len(ys) == 4:
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                value_height = int((y2 - y1) * 1.5)
                layout_dict[label] = {
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2 + value_height
                }
def compute_form_bounds(form_layout):
    coords = []
    for box in form_layout.values():
        if all(k in box for k in ("x1", "y1", "x2", "y2")):
            coords.extend([
                (box["x1"], box["y1"]),
                (box["x2"], box["y2"])
            ])
    if not coords:
        return None
    xs, ys = zip(*coords)
    return min(xs), min(ys), max(xs), max(ys)
