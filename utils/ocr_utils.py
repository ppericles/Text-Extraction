def normalize(text):
    return (
        text.upper().strip()
        .replace("Ά", "Α").replace("Έ", "Ε").replace("Ή", "Η")
        .replace("Ί", "Ι").replace("Ό", "Ο").replace("Ύ", "Υ")
        .replace("Ώ", "Ω").replace("Ϊ", "Ι").replace("Ϋ", "Υ")
    )

def detect_header_regions(blocks):
    # Example logic to find likely header positions
    headers = [b for b in blocks if "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ" in normalize(b["text"])]
    if headers:
        return headers[0]["center"][1]  # Return Y position of first header
    return None

def compute_form_bounds(blocks):
    # Computes form bounding box from OCR blocks
    xs = [b["center"][0] for b in blocks]
    ys = [b["center"][1] for b in blocks]
    return min(xs), min(ys), max(xs), max(ys)
