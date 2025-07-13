def get_form_bounding_box(form_boxes):
    coords = []
    for box in form_boxes.values():
        if all(k in box for k in ("x1", "y1", "x2", "y2")):
            coords.extend([
                (box["x1"], box["y1"]),
                (box["x2"], box["y2"])
            ])
    if not coords:
        return None
    xs, ys = zip(*coords)
    return min(xs), min(ys), max(xs), max(ys)
