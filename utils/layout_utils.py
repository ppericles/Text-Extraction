def get_form_bounding_box(layout):
    x1s = [box["x1"] for box in layout.values()]
    y1s = [box["y1"] for box in layout.values()]
    x2s = [box["x2"] for box in layout.values()]
    y2s = [box["y2"] for box in layout.values()]
    return min(x1s), min(y1s), max(x2s), max(y2s)
