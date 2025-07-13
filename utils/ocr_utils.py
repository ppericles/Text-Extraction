
from unidecode import unidecode

def normalize(text):
    return unidecode(text.upper().strip())

def detect_header_regions(annotations, field_labels, layout_dict):
    # Draw the 8 auto-detected header boxes
    for label in field_labels:
        box = field_boxes.get(label)
        if box and all(k in box for k in ("x1", "y1", "x2", "y2")):
            x1, y1 = box["x1"], box["y1"]
            x2, y2 = box["x2"], box["y2"]
            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
            draw.text((x1, y1 - 12), label, fill="green")
            
    normalized_labels = {normalize(lbl): lbl for lbl in field_labels}
    for ann in annotations[1:]:
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
