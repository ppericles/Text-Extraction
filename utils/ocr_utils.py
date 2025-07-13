from difflib import get_close_matches
from unidecode import unidecode
import streamlit as st

def normalize(text):
    return unidecode(text.upper().strip())

def group_lines(annotations, y_thresh=25):
    lines = []
    for ann in annotations[1:]:  # Skip full block
        text = ann.description
        vertices = ann.bounding_poly.vertices
        if not vertices:
            continue
        ys = [v.y for v in vertices if v.y is not None]
        x_center = sum(v.x for v in vertices if v.x is not None) / len(vertices)
        y_center = sum(ys) / len(ys) if ys else 0

        placed = False
        for line in lines:
            avg_y = line["avg_y"]
            if abs(y_center - avg_y) < y_thresh:
                line["words"].append((x_center, text, vertices))
                line["avg_y"] = (line["avg_y"] * len(line["words"]) + y_center) / (len(line["words"]) + 1)
                placed = True
                break

        if not placed:
            lines.append({"avg_y": y_center, "words": [(x_center, text, vertices)]})
    return lines

def detect_header_regions(annotations, field_labels, layout_dict, debug=True, cutoff=0.75):
    normalized_targets = [normalize(lbl) for lbl in field_labels]
    hits = []
    match_log = []

    lines = group_lines(annotations)
    all_line_texts = []

    for line in lines:
        words = sorted(line["words"], key=lambda w: w[0])
        full_text = " ".join(word[1] for word in words)
        all_line_texts.append(full_text)

        normalized_line = normalize(full_text)
        match = get_close_matches(normalized_line, normalized_targets, n=1, cutoff=cutoff)
        if match:
            label = field_labels[normalized_targets.index(match[0])]
            match_log.append(f"✔️ '{full_text}' → '{label}'")

            all_x = []
            all_y = []
            for _, _, vertices in words:
                all_x.extend([v.x for v in vertices if v.x is not None])
                all_y.extend([v.y for v in vertices if v.y is not None])
            if all_x and all_y:
                x1, x2 = min(all_x), max(all_x)
                y1, y2 = min(all_y), max(all_y)
                value_height = int((y2 - y1) * 1.5)
                layout_dict[label] = {
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2 + value_height
                }
                hits.append(label)
            else:
                match_log.append(f"⚠️ Incomplete bounding box for '{label}'")

    missing = [lbl for lbl in field_labels if lbl not in hits]
    if missing:
        match_log.append(f"🚫 Labels not detected: {missing}")

    if debug:
        st.subheader("🧪 OCR Text Lines")
        st.code("\n".join(all_line_texts))
        st.subheader("📌 Header Detection Matches")
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
