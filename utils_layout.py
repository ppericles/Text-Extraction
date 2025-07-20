# ==== utils_layout.py ====

import json

class LayoutManager:
    def __init__(self, image_size):
        self.width, self.height = image_size

    def to_normalized(self, box):
        """
        Convert absolute box (x1, y1, x2, y2) to normalized coordinates.
        """
        x1, y1, x2, y2 = box
        return [
            round(x1 / self.width, 4),
            round(y1 / self.height, 4),
            round(x2 / self.width, 4),
            round(y2 / self.height, 4)
        ]

    def to_absolute(self, box):
        """
        Convert normalized box to absolute pixel coordinates.
        """
        x1, y1, x2, y2 = box
        return [
            int(x1 * self.width),
            int(y1 * self.height),
            int(x2 * self.width),
            int(y2 * self.height)
        ]

    def save_layout(self, layout_dict):
        """
        Normalize all boxes in layout_dict.
        """
        return {
            label: self.to_normalized(tuple(box))
            for label, box in layout_dict.items()
        }

    def load_layout(self, layout_dict):
        """
        Convert normalized layout to absolute coordinates.
        """
        return {
            label: self.to_absolute(box)
            for label, box in layout_dict.items()
        }


def load_default_layout(zone_id, template_paths):
    """
    Load default layout template for a given zone.
    """
    path = template_paths.get(zone_id)
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_zone_layout(zone_id, expected_labels, layout_managers, box_layouts, st):
    """
    Validate that all expected labels exist in the layout.
    """
    missing = [label for label in expected_labels if label not in box_layouts[zone_id]]
    if missing:
        st.warning(f"⚠️ Zone {zone_id} is missing fields: {', '.join(missing)}")
    else:
        st.success(f"✅ Zone {zone_id} layout includes all expected fields.")
