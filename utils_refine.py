# =============================================================================
# FILE: utils_refine.py
# VERSION: 1.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Unified layout refinement pipeline.
#              Splits master and detail zones, labels subzones by OCR density,
#              and exports final layout as JSON.
# =============================================================================

import json
from PIL import ImageDraw, ImageFont
import streamlit as st

def refine_layout_with_zones(layout, boxes, image, manual=False, form_id=""):
    w, h = image.size
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    def split_and_label(zone_key, label_a, label_b, color_a, color_b, slider_key):
        zone = layout.get(zone_key)
        if not zone or len(zone) != 4:
            return

        x1, y1, x2, y2 = zone
        split_y = st.slider(f"{zone_key} Split Y", y1 + 0.05, y2 - 0.05, (y1 + y2) / 2, 0.01, key=f"{form_id}_{slider_key}") if manual else (y1 + y2) / 2

        top_zone = [x1, y1, x2, split_y]
        bottom_zone = [x1, split_y, x2, y2]

        def count_boxes(zone):
            zx1, zy1, zx2, zy2 = [int(coord * dim) for coord, dim in zip(zone, (w, h, w, h))]
            return sum(1 for bx1, by1, bx2, by2 in boxes if bx1 >= zx1 and bx2 <= zx2 and by1 >= zy1 and by2 <= zy2)

        top_count = count_boxes(top_zone)
        bottom_count = count_boxes(bottom_zone)

        layout[label_a] = top_zone if top_count >= bottom_count else bottom_zone
        layout[label_b] = bottom_zone if top_count >= bottom_count else top_zone

        for label, color in [(label_a, color_a), (label_b, color_b)]:
            box = layout[label]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, (w, h, w, h))]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 5, y1 + 5), label.replace("_box", "").upper(), fill=color, font=font)

    split_and_label("master_box", "group_a_box", "group_b_box", "blue", "red", "master_split")
    split_and_label("detail_box", "detail_top_box", "detail_bottom_box", "purple", "orange", "detail_split")

    return layout, overlay

def export_layout_json(layout, form_id):
    layout_json = json.dumps(layout, indent=2)
    st.download_button(
        label="💾 Download Layout JSON",
        data=layout_json,
        file_name=f"{form_id}_layout.json",
        mime="application/json"
    )
