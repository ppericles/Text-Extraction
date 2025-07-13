
def handle_click(coords, width, height, field_label, field_boxes, session):
    x, y = coords["x"], coords["y"]
    if not (0 <= x < width and 0 <= y < height):
        return "outside"

    if session.click_stage == "start":
        field_boxes[field_label] = {"x1": x, "y1": y}
        session.click_stage = "end"
        return "start"
    else:
        if field_label not in field_boxes:
            field_boxes[field_label] = {}
        field_boxes[field_label].update({"x2": x, "y2": y})
        session.click_stage = "start"
        return "end"
