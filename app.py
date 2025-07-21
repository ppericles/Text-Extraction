# ==== FILE: app.py ====
# Version: 1.9.0
# Author: Pericles & Copilot

import streamlit as st
from PIL import Image, ImageDraw
import os, json, tempfile
import cv2, numpy as np

from utils_ocr import form_parser_ocr, match_fields_with_fallback, vision_api_ocr
from utils_image import optimize_image, resize_for_preview, trim_whitespace, split_zones_fixed

# ==== Column Detection ====

def suggest_column_breaks(pil_image, threshold=220, min_gap=15):
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    profile = np.sum(binary, axis=0)
    gaps = np.where(profile < np.max(profile) * 0.1)[0]
    separators, prev = [], None
    for idx in gaps:
        if prev is None or idx - prev > min_gap:
            separators.append(idx)
        prev = idx
    columns = []
    for i in range(len(separators) - 1):
        x1, x2 = separators[i] / binary.shape[1], separators[i + 1] / binary.shape[1]
        if x2 - x1 > 0.05:
            columns.append((x1, x2))
    return columns

def extract_table(table_img, config, column_breaks, rows=10, header=True):
    w, h = table_img.size
    row_h = h / (rows + int(header))
    headers, data = [], []
    for r in range(rows + int(header)):
        y1, y2 = int(r * row_h), int((r + 1) * row_h)
        row = {}
        for i, (x1, x2) in enumerate(column_breaks):
            cell = table_img.crop((int(x1 * w), y1, int(x2 * w), y2))
            text = vision_api_ocr(cell).strip()
            if header and r == 0:
                headers.append(text or f"col_{i}")
            elif r >= int(header):
                key = headers[i] if i < len(headers) else f"col_{i}"
                row[key] = text
        if r >= int(header):
            data.append(row)
    return headers, data

def process_single_form(image, box, index, config, layout):
    w, h = image.size
    x1, y1, x2, y2 = box
    form_crop = image.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))
    zones, _ = split_zones_fixed(form_crop, layout.get("master_ratio", 0.5))
    master_zone, detail_zone = zones

    gw, gh = master_zone.size
    xa1, ya1, xa2, ya2 = layout.get("group_a_box", [0.0, 0.0, 0.2, 1.0])
    xb1, yb1, xb2, yb2 = layout.get("group_b_box", [0.2, 0.0, 1.0, 0.5])
    group_a = master_zone.crop((int(xa1*gw), int(ya1*gh), int(xa2*gw), int(ya2*gh)))
    group_b = master_zone.crop((int(xb1*gw), int(yb1*gh), int(xb2*gw), int(yb2*gh)))

    dw, dh = detail_zone.size
    xt1, yt1, xt2, yt2 = layout.get("detail_box", [0.0, 0.0, 1.0, 1.0])
    detail_crop = detail_zone.crop((int(xt1*dw), int(yt1*dh), int(xt2*dw), int(yt2*dh)))

    expected_a = ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"]
    expected_b = ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]

    fields_a = form_parser_ocr(group_a, **config)
    fields_b = form_parser_ocr(group_b, **config)

    matched_a = match_fields_with_fallback(expected_a, fields_a, group_a, layout)
    matched_b = match_fields_with_fallback(expected_b, fields_b, group_b, layout)

    meridos = matched_a.get("ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", {}).get("value", "")
    if not meridos and index > 0:
        prev = layout.get(f"form_{index-1}_meridos", "")
        if prev.isdigit():
            meridos = str(int(prev) + 1)
            matched_a["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"] = {"value": meridos, "confidence": 0}
    layout[f"form_{index}_meridos"] = meridos

    column_breaks = layout.get("table_columns")
    if layout.get("auto_detect", True):
        column_breaks = suggest_column_breaks(detail_crop)

    _, table_rows = extract_table(detail_crop, config, column_breaks)

    return {
        "master": master_zone,
        "detail": detail_zone,
        "group_a": matched_a,
        "group_b": matched_b,
        "table_crop": detail_crop,
        "column_breaks": column_breaks,
        "table_rows": table_rows
    }
  # ==== UI ====

st.set_page_config(page_title="📄 Registry Parser", layout="wide")
st.title("📄 Registry Form Parser")

# === Setup ===
CONFIG_PATH = "config/processor_config.json"
os.makedirs("config", exist_ok=True)
default_config = {"project_id": "", "location": "", "processor_id": ""}
if os.path.exists(CONFIG_PATH):
    default_config = json.load(open(CONFIG_PATH))

# === Sidebar ===
st.sidebar.markdown("### 🔐 Credentials")
cred_file = st.sidebar.file_uploader("Upload Google JSON", type="json")
if cred_file:
    temp_path = tempfile.NamedTemporaryFile(delete=False)
    temp_path.write(cred_file.read()), temp_path.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path.name
    st.sidebar.success("✅ Loaded")

st.sidebar.markdown("### ⚙️ Document AI")
project_id = st.sidebar.text_input("Project ID", value=default_config["project_id"])
location = st.sidebar.text_input("Location", value=default_config["location"])
processor_id = st.sidebar.text_input("Processor ID", value=default_config["processor_id"])

if st.sidebar.button("💾 Save Config"):
    json.dump({
        "project_id": project_id.strip(),
        "location": location.strip(),
        "processor_id": processor_id.strip()
    }, open(CONFIG_PATH, "w"))
    st.sidebar.success("✅ Saved")

# === Upload Images ===
files = st.file_uploader("📤 Upload Registry Scans", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if files and all([project_id, location, processor_id]):
    config = {"project_id": project_id, "location": location, "processor_id": processor_id}

    for file in files:
        st.header(f"📄 `{file.name}`")
        img = trim_whitespace(optimize_image(Image.open(file)))
        st.image(resize_for_preview(img), caption="📄 Full Image", use_column_width=True)

        form_boxes = []
        for i in range(3):
            with st.expander(f"📐 Form {i+1} Bounding Box"):
                x1 = st.slider("X1", 0.0, 1.0, 0.0, key=f"x1_{i}")
                y1 = st.slider("Y1", 0.0, 1.0, i * 0.33, key=f"y1_{i}")
                x2 = st.slider("X2", 0.0, 1.0, 1.0, key=f"x2_{i}")
                y2 = st.slider("Y2", 0.0, 1.0, (i + 1) * 0.33, key=f"y2_{i}")
                form_boxes.append((x1, y1, x2, y2))

        for i, box in enumerate(form_boxes):
            st.subheader(f"🔍 Form {i+1} Results")

            auto = st.checkbox("Auto-detect table columns", value=True, key=f"auto_{i}")
            layout = {
                "master_ratio": 0.5,
                "group_a_box": [0.0, 0.0, 0.2, 1.0],
                "group_b_box": [0.2, 0.0, 1.0, 0.5],
                "detail_box": [0.0, 0.0, 1.0, 1.0],
                "auto_detect": auto
            }

            if not auto:
                st.markdown("📐 Define Table Columns")
                table_columns = []
                for c in range(6):
                    cx1 = st.slider(f"Column {c+1} - X1", 0.0, 1.0, c * 0.15, key=f"cx1_{i}_{c}")
                    cx2 = st.slider(f"Column {c+1} - X2", 0.0, 1.0, (c + 1) * 0.15, key=f"cx2_{i}_{c}")
                    table_columns.append((cx1, cx2))
                layout["table_columns"] = table_columns

            result = process_single_form(img, box, i, config, layout)

            st.image(resize_for_preview(result["master"]), caption="🟦 Master Zone", use_column_width=True)

            # Column overlay
            overlay = result["detail"].copy()
            draw = ImageDraw.Draw(overlay)
            w, h = overlay.size
            for x1, _ in result["column_breaks"]:
                x = int(x1 * w)
                draw.line([(x, 0), (x, h)], fill="red", width=2)
            st.image(resize_for_preview(overlay), caption="📐 Table Column Breaks", use_column_width=True)

            st.markdown("### 🧾 Group A (ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ)")
            for label, data in result["group_a"].items():
                emoji = "🟢" if data["confidence"] >= 90 else "🟡" if data["confidence"] >= 70 else "🔴"
                st.text(f"{emoji} {label}: {data['value']} ({data['confidence']}%)")

            st.markdown("### 🧾 Group B")
            for label, data in result["group_b"].items():
                emoji = "🟢" if data["confidence"] >= 90 else "🟡" if data["confidence"] >= 70 else "🔴"
                st.text(f"{emoji} {label}: {data['value']} ({data['confidence']}%)")

            st.markdown("### 📊 Parsed Table Rows")
            if result["table_rows"]:
                st.dataframe(result["table_rows"], use_container_width=True)
            else:
                st.warning("⚠️ No table rows extracted.")  
