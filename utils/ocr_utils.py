from difflib import get_close_matches
from unidecode import unidecode
import streamlit as st

def normalize(text):
    return unidecode(text.upper().strip())

def detect_header_regions(annotations, field_labels, layout_dict, debug=False):
    normalized_targets = [normalize(lbl) for lbl in field_labels]
    hits = []
    match_log = []  # For displaying in Streamlit

    for ann in annotations[1:]:
        txt_raw = ann.description
        txt = normalize(txt_raw)
        match = get_close_matches(txt, normalized_targets, n=1, cutoff=0.8)
        if match:
            label = field_labels[normalized_targets.index(match[0])]
            match_log.append(f"✔️ '{txt_raw}' → '{label}'")

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
                hits.append(label)
            else:
                match_log.append(f"⚠️ Bounding box incomplete for '{label}'")

    missing = [lbl for lbl in field_labels if lbl not in hits]
    if missing:
        match_log.append(f"🚫 Labels not detected: {missing}")

    if debug and match_log:
        st.subheader("🧪 Header Detection Log")
        st.code("\n".join(match_log))
        
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
