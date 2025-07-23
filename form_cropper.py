# =============================================================================
# FILE: image_cropper.py
# VERSION: 1.1.0
# AUTHOR: Pericles & Copilot
# DESCRIPTION: Interactive cropping tool for scanned registry forms.
#              Supports canvas-based selection, preview, confirmation,
#              undo, zone overlays with color-coded bounding boxes,
#              and toggleable visibility per zone.
# =============================================================================

from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from utils_image import resize_for_preview, draw_box

# === Zone Colors ===
ZONE_COLORS = {
    "master": "rgba(0, 255, 0, 0.2)",     # green
    "group_a": "rgba(0, 0, 255, 0.2)",    # blue
    "group_b": "rgba(255, 0, 0, 0.2)",    # red
    "detail": "rgba(128, 0, 128, 0.2)"    # purple
}

def crop_and_confirm_forms(image: Image.Image, max_crops=5, key_prefix="crop"):
    """
    Interactive cropping tool for extracting form regions from a scanned image.
    Supports preview, confirmation, undo, and zone overlays with toggleable visibility.
    """
    st.markdown("### ✂️ Crop Forms")
    canvas_height = image.height
    canvas_width = image.width

    if f"{key_prefix}_history" not in st.session_state:
        st.session_state[f"{key_prefix}_history"] = []

    for i in range(max_crops):
        crop_key = f"{key_prefix}_{i}"
        st.markdown(f"#### Crop Form {i+1}")

        canvas_result = st_canvas(
            background_image=image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            fill_color=ZONE_COLORS["master"],
            stroke_width=2,
            key=f"{crop_key}_canvas"
        )

        preview_crop = None
        if canvas_result.json_data and "objects" in canvas_result.json_data:
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    x1 = int(obj["left"])
                    y1 = int(obj["top"])
                    x2 = int(obj["left"] + obj["width"])
                    y2 = int(obj["top"] + obj["height"])
                    preview_crop = image.crop((x1, y1, x2, y2))
                    st.image(resize_for_preview(preview_crop), caption=f"🖼️ Preview Crop {i+1}", use_column_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"✅ Confirm Crop {i+1}", key=f"{crop_key}_confirm") and preview_crop:
                st.session_state[f"{key_prefix}_history"].append(preview_crop)
                st.success(f"Form {i+1} confirmed.")
        with col2:
            if st.button(f"↩️ Undo Last", key=f"{crop_key}_undo"):
                if st.session_state[f"{key_prefix}_history"]:
                    st.session_state[f"{key_prefix}_history"].pop()
                    st.warning("Last crop removed.")

    return st.session_state[f"{key_prefix}_history"]

def draw_zone_overlay(image: Image.Image, layout: dict, form_id: str):
    """
    Draws zone overlays on a cropped form image with toggleable visibility.
    """
    st.markdown("### 🧭 Zone Visibility")

    show_master = st.checkbox("🟩 Show Master Zone", value=True, key=f"{form_id}_toggle_master")
    show_group_a = st.checkbox("🟦 Show Group A", value=True, key=f"{form_id}_toggle_group_a")
    show_group_b = st.checkbox("🟥 Show Group B", value=True, key=f"{form_id}_toggle_group_b")
    show_detail = st.checkbox("🟪 Show Detail Zone", value=True, key=f"{form_id}_toggle_detail")

    overlay = image.copy()

    if show_master and "master_box" in layout:
        draw_box(overlay, layout["master_box"], fill=ZONE_COLORS["master"])
    if show_group_a and "group_a_box" in layout:
        draw_box(overlay, layout["group_a_box"], fill=ZONE_COLORS["group_a"])
    if show_group_b and "group_b_box" in layout:
        draw_box(overlay, layout["group_b_box"], fill=ZONE_COLORS["group_b"])
    if show_detail and "detail_box" in layout:
        draw_box(overlay, layout["detail_box"], fill=ZONE_COLORS["detail"])

    st.image(resize_for_preview(overlay), caption="🔍 Zone Overlay", use_column_width=True)
