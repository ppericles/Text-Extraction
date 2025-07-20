# ==== utils_layout.py ====

import json, os
from PIL import ImageDraw

EXPORT_DIR = "exports/layout_versions"
INDEX_FILE = os.path.join(EXPORT_DIR, "layout_index.json")

# 📐 Convert box between pixel ↔ normalized coordinates
def convert_box(box, image_size, to_normalized=True):
    if not box or any(v is None for v in box): return (None, None, None, None)
    x, y, w, h = box
    iw, ih = image_size
    return (
        x / iw, y / ih, w / iw, h / ih
    ) if to_normalized else (
        x * iw, y * ih, w * iw, h * ih
    )

# 🧭 LayoutManager class for pixel↔normalized handling
class LayoutManager:
    def __init__(self, image_size): self.image_size = image_size
    def to_pixel(self, box): return convert_box(box, self.image_size, False)
    def to_normalized(self, box): return convert_box(box, self.image_size, True)
    def load_layout(self, layout_dict): return {label: self.to_pixel(box) for label, box in layout_dict.items()}
    def save_layout(self, layout_dict): return {label: self.to_normalized(box) for label, box in layout_dict.items()}

# 🟨 Detect missing layout fields
def get_missing_layout_fields(layout_pixels, required_labels):
    return [
        label for label in required_labels
        if label not in layout_pixels or any(v is None for v in layout_pixels[label])
    ]

# ✏️ Editable layout editor
def ensure_zone_layout(zid, expected_labels, layout_managers, box_layouts, st):
    st.subheader(f"🛠️ Layout Editor for Zone {zid}")
    manager = layout_managers[zid]
    layout_pixels = manager.load_layout(box_layouts.get(zid, {}))

    editor_rows = []
    for label in expected_labels:
        box = layout_pixels.get(label, (None, None, None, None))
        x, y, w, h = box
        editor_rows.append({"Label": label, "X": x, "Y": y, "Width": w, "Height": h})

    editor_df = st.data_editor(
        editor_rows,
        use_container_width=True,
        num_rows="dynamic",
        key=f"layout_editor_{zid}"
    )

    edited_layout = {
        row["Label"]: (row["X"], row["Y"], row["Width"], row["Height"])
        for row in editor_df
        if all(v is not None for v in (row["X"], row["Y"], row["Width"], row["Height"]))
    }

    if edited_layout:
        box_layouts[zid] = manager.save_layout(edited_layout)
        st.success(f"✅ Layout updated for Zone {zid}")
        register_layout_version(zid, box_layouts[zid], version="manual_edit")
    else:
        st.warning(f"⚠️ Layout still incomplete for Zone {zid}")

# 🖼️ Visual layout overlay
def draw_layout_overlay(zone_img, layout_pixels, box_color="red"):
    img = zone_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for label, (x, y, w, h) in layout_pixels.items():
        rect = [(x, y), (x + w, y + h)]
        draw.rectangle(rect, outline=box_color, width=2)
        draw.text((x, max(0, y - 20)), label, fill=box_color)
    return img

# 🧱 Default template loader
def load_default_layout(zid, template_store):
    return template_store.get(zid, {})

# 💾 Save layout snapshot to exports + register
def register_layout_version(zid, layout_data, version="v1"):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, f"zone_{zid}_{version}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(layout_data, f, indent=2, ensure_ascii=False)
    
    # Update index
    index = {}
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            try: index = json.load(f)
            except: pass

    index.setdefault(zid, []).append({
        "version": version,
        "file": path,
    })

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

# 📚 Load all registered versions
def get_registered_layout_versions(zid=None):
    if not os.path.exists(INDEX_FILE): return {}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)
    return index if zid is None else index.get(zid, [])
