# ==== START OF PART 1: Imports & Utilities ====

import streamlit as st
import os, json, re, unicodedata
from PIL import Image, ImageDraw
from io import BytesIO
from datetime import datetime
import pandas as pd
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

# 🔡 Text Normalization Functions
# (fix_latin_greek, fix_cyrillic_greek, normalize)

# 📅 Date Normalization
def normalize_date(text):
    text = text.strip()
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y"]:
        try:
            return datetime.strptime(text, fmt).strftime("%d/%m/%Y")
        except:
            continue
    return text

# 🛡️ Registry Field Validation
# 🧠 Suggestion Generation
# ✂️ Image Preprocessing (trim_whitespace, crop_left, split_zones_fixed)

# 📐 Box Conversion + OCR Confidence
# 🩹 Vision OCR Fallback Function
# 🧠 Document AI Parsing Function

# ==== END OF PART 1 ====
# ==== START OF PART 2: Layout & App Initialization ====

# 📐 LayoutManager class
class LayoutManager:
    def __init__(self, image_size): self.image_size = image_size
    def to_pixel(self, box): return convert_box(box, self.image_size, to_normalized=False)
    def to_normalized(self, box): return convert_box(box, self.image_size, to_normalized=True)
    def load_layout(self, layout_dict): return {label: self.to_pixel(box) for label, box in layout_dict.items()}
    def save_layout(self, layout_dict): return {label: self.to_normalized(box) for label, box in layout_dict.items()}

# ✏️ ensure_zone_layout helper
def ensure_zone_layout(zid, expected_labels, layout_managers, box_layouts):
    st.subheader(f"🛠️ Layout Editor for Zone {zid}")
    manager = layout_managers[zid]
    layout_pixels = manager.load_layout(box_layouts.get(zid, {}))
    editor_rows = []
    for label in expected_labels:
        box = layout_pixels.get(label, (None, None, None, None))
        x, y, w, h = box
        editor_rows.append({"Label": label, "X": x, "Y": y, "Width": w, "Height": h})
    editor_df = st.data_editor(pd.DataFrame(editor_rows), use_container_width=True, num_rows="dynamic", key=f"layout_editor_{zid}")
    edited_layout = {
        row["Label"]: (row["X"], row["Y"], row["Width"], row["Height"])
        for _, row in editor_df.iterrows()
        if all(v is not None for v in (row["X"], row["Y"], row["Width"], row["Height"]))
    }
    if edited_layout:
        box_layouts[zid] = manager.save_layout(edited_layout)
        st.success(f"✅ Saved layout for Zone {zid}")
    else:
        st.warning(f"⚠️ Layout still incomplete for Zone {zid}")

# 🚀 Streamlit Setup
st.set_page_config(page_title="📜 Registry Parser", layout="wide")
st.title("📜 Greek Registry Parser — Master & Detail Mapping")

# 🎯 Metadata Fields
master_field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ",
    "ΕΠΩΝΥΜΟΝ",
    "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΚΥΡΙΟΝ ΟΝΟΜΑ",
    "ΗΜΕΡΟΜΗΝΙΑ ΓΕΝΝΗΣΕΩΣ",
    "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ",
    "ΔΙΕΥΘΥΝΣΗ"
]

# 📦 Data Containers
metadata_rows = []
detail_rows = []
box_layouts = {}

# 🎛️ Sidebar Controls
st.sidebar.header("⚙️ Parser Settings")
overlap = st.sidebar.slider("🔁 Zone Overlap (px)", 0, 120, value=40)

# 🔐 GCP Credentials
cred_file = st.sidebar.file_uploader("🔐 Upload Google Credentials (.json)", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ GCP credentials loaded")
    project_id = "heroic-gantry-380919"
    processor_id = "8f7f56e900fbb37e"
    location = "eu"

# 📥 Optional Layout Import
layout_file = st.sidebar.file_uploader("📥 Import Box Layouts (.json)", type=["json"])
if layout_file:
    try:
        box_layouts = json.load(layout_file)
        st.sidebar.success(f"📦 Loaded layouts for {len(box_layouts)} zones")
    except Exception as e:
        st.sidebar.error(f"❌ Layout import error: {e}")

# ==== END OF PART 2 ====
# ==== START OF PART 3: Image Upload & Zone Preview ====

# 📄 Upload Registry Scan
uploaded_image = st.file_uploader("📄 Upload Registry Scan", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("📎 Upload a registry page to begin.")
    st.stop()

# ✂️ Preprocessing and Segmentation
try:
    original = Image.open(uploaded_image)
    cropped = crop_left(trim_whitespace(original))
    zones, bounds = split_zones_fixed(cropped, overlap)
    if not zones:
        st.error("🚫 Zone segmentation failed.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error during image preprocessing: {e}")
    st.stop()

# 📐 LayoutManager Initialization per Zone
layout_managers = {
    str(i + 1): LayoutManager(zones[i].size)
    for i in range(len(zones))
}

# 🔢 Zone Role Mapping
zone_roles = {
    "1": "🔷 Master Section (Top)",
    "2": "🆔 Form ID Section (ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ)",
    "3": "📊 Detail Table Section"
}

# 🖼️ Preview Cropped Image
st.image(cropped, caption="📌 Cropped Registry Page (Left Side)", use_column_width=True)

# 🗂️ Preview All Zones
st.subheader("📌 Zone Structure Overview")
for i, zone in enumerate(zones, start=1):
    role = zone_roles.get(str(i), "Unknown")
    st.image(zone, caption=f"Zone {i} — {role}", width=300)

# ==== END OF PART 3 ====
# ==== START OF PART 4: Zone Processing Loop ====

for idx, zone in enumerate(zones, start=1):
    zid = str(idx)
    manager = layout_managers[zid]
    st.header(f"{zone_roles.get(zid, f'📄 Zone {zid}')} — Zone {zid}")

    # 🧭 Define expected labels per zone
    expected_labels = (
        ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"] if zid == "2"
        else [l for l in master_field_labels if l != "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"]
    )

    # ✏️ Live-edit layout if missing
    ensure_zone_layout(zid, expected_labels, layout_managers, box_layouts)

    # 🧠 Run Document AI on current zone
    doc = parse_docai(zone.copy(), project_id, processor_id, location)
    full_text = doc.text if doc else ""

    # 🔷 Zone 1: Master fields except ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ
    if zid == "1":
        field_map = {label: {"Corrected": "", "Confidence": 0.0, "Issues": [], "Suggestion": None}
                     for label in expected_labels}
        if doc:
            for page in doc.pages:
                for f in page.form_fields:
                    label_raw = extract_text_from_anchor(f.field_name.text_anchor, full_text)
                    value_raw = extract_text_from_anchor(f.field_value.text_anchor, full_text)
                    label_norm = normalize(label_raw)
                    if label_norm in expected_labels:
                        corrected = normalize(value_raw)
                        conf = round(f.field_value.confidence * 100, 2)
                        issues = validate_registry_field(label_norm, corrected, conf)
                        suggestion = suggest_fix(label_norm, corrected, issues)
                        field_map[label_norm] = {"Corrected": corrected, "Confidence": conf, "Issues": issues, "Suggestion": suggestion}

        # 🩹 Vision fallback for missing fields
        for label in expected_labels:
            if not field_map[label]["Corrected"]:
                box = box_layouts[zid].get(label)
                if box:
                    raw, conf = extract_field_from_box_with_vision(zone, box, label)
                    corrected = normalize(raw)
                    issues = validate_registry_field(label, corrected, conf)
                    suggestion = suggest_fix(label, corrected, issues)
                    field_map[label] = {"Corrected": corrected, "Confidence": conf, "Issues": issues, "Suggestion": suggestion}

        # ✅ Store metadata row (FormID not set yet)
        metadata_rows.append({
            "FormID": f"ZONE_{zid}",
            **{label: field_map[label]["Corrected"] for label in expected_labels}
        })

    # 🆔 Zone 2: Extract ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ
    elif zid == "2":
        label = "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"
        field_map = {
            label: {"Corrected": "", "Confidence": 0.0, "Issues": [], "Suggestion": None}
        }
        if doc:
            for page in doc.pages:
                for f in page.form_fields:
                    label_raw = extract_text_from_anchor(f.field_name.text_anchor, full_text)
                    value_raw = extract_text_from_anchor(f.field_value.text_anchor, full_text)
                    label_norm = normalize(label_raw)
                    if label_norm == label:
                        corrected = normalize(value_raw)
                        conf = round(f.field_value.confidence * 100, 2)
                        issues = validate_registry_field(label_norm, corrected, conf)
                        suggestion = suggest_fix(label_norm, corrected, issues)
                        field_map[label] = {"Corrected": corrected, "Confidence": conf, "Issues": issues, "Suggestion": suggestion}

        # 🩹 Vision fallback
        if not field_map[label]["Corrected"]:
            box = box_layouts[zid].get(label)
            if box:
                raw, conf = extract_field_from_box_with_vision(zone, box, label)
                corrected = normalize(raw)
                issues = validate_registry_field(label, corrected, conf)
                suggestion = suggest_fix(label, corrected, issues)
                field_map[label] = {"Corrected": corrected, "Confidence": conf, "Issues": issues, "Suggestion": suggestion}

        # 🆔 Use ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ as true FormID
        form_id = field_map[label]["Corrected"] or f"ZONE_{zid}"
        st.markdown(f"🆔 Extracted FormID from Zone 2: **`{form_id}`**")

    # 📊 Zone 3: Parse Detail Table Rows
    elif zid == "3" and doc:
        expected_columns = [
            "Α/Α",
            "ΤΟΜΟΣ",
            "ΑΡΙΘ.",
            "ΗΜΕΡΟΜΗΝΙΑ ΜΕΤΑΓΡΑΦΗΣ",
            "ΑΡΙΘ. ΕΓΓΡΑΦΟΥ\nΚΑΙ ΕΤΟΣ ΑΥΤΟΥ",
            "ΣΥΜΒΟΛΑΙΟΓΡΑΦΟΣ\nΉ Η ΕΚΔΟΥΣΑ ΑΡΧΗ",
            "ΕΙΔΟΣ ΑΚΙΝΗΤΟΥ",
            "ΚΑΤΟΙΚΙΑ"
        ]

        def match_column_label(text):
            norm = normalize(text.replace("\n", " ").strip())
            for expected in expected_columns:
                norm_expected = normalize(expected.replace("\n", " ").strip())
                if norm_expected in norm or norm in norm_expected:
                    return expected
            return None

        for page in doc.pages:
            for table in page.tables:
                headers = []
                for header_row in table.header_rows:
                    for cell in header_row.cells:
                        raw = extract_text_from_anchor(cell.layout.text_anchor, full_text)
                        matched = match_column_label(raw)
                        if matched and matched not in headers:
                            headers.append(matched)
                        if len(headers) >= 6:
                            break

                if len(headers) < 6:
                    st.warning("⚠️ Table header not matched — skipping table")
                    continue

                active_headers = headers[:6]
                st.markdown(f"📑 Registry Table — Columns: `{', '.join(active_headers)}`")

                for row in table.body_rows:
                    row_data = {"FormID": form_id}
                    for i in range(6):
                        key = active_headers[i] if i < len(active_headers) else f"COL_{i}"
                        cell = row.cells[i] if i < len(row.cells) else None
                        value = extract_text_from_anchor(cell.layout.text_anchor, full_text) if cell else ""
                        if "ΗΜΕΡ" in key:
                            value = normalize_date(value)
                        row_data[key] = normalize(value)
                    detail_rows.append(row_data)

        # 🖼️ Visual linkage preview between FormID zone and registry table zone
        st.subheader("🖼️ FormID & Table Zone Linkage")
        col1, col2 = st.columns(2)
        with col1:
            st.image(zones[1], caption="Zone 2 — ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", use_column_width=True)
            st.markdown(f"🆔 FormID: **`{form_id}`**")
        with col2:
            st.image(zones[2], caption="Zone 3 — Registry Table", use_column_width=True)
            st.markdown(f"📎 Linked Detail Rows: `{len(detail_rows)}` rows")

# ==== END OF PART 4 ====
# ==== START OF PART 5: Review, Export & Audit ====

# 🧠 Metadata Review UI
st.header("📊 Metadata Review & Final Corrections")
auto_apply = st.checkbox("💡 Auto-apply Suggestions", value=False)

final_metadata = []
for row in metadata_rows:
    fid = row["FormID"]
    st.subheader(f"📄 FormID: {fid}")
    corrected_row = {"FormID": fid}
    for label in master_field_labels:
        if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": continue
        val = row.get(label, "")
        final = st.text_input(f"{label}", value=val, key=f"{fid}_{label}")
        corrected_row[label] = final
    final_metadata.append(corrected_row)

# 📥 Export Master Metadata
st.header("📤 Export Master Metadata")
df_master = pd.DataFrame(final_metadata)
st.dataframe(df_master, use_container_width=True)
st.download_button("📄 Download Metadata CSV", df_master.to_csv(index=False), "metadata_master.csv", mime="text/csv")
st.download_button("📄 Download Metadata JSON", json.dumps(final_metadata, indent=2, ensure_ascii=False), "metadata_master.json", mime="application/json")

# 📥 Export Registry Detail Table
st.header("📤 Export Registry Detail Table")
df_detail = pd.DataFrame(detail_rows)
st.dataframe(df_detail, use_container_width=True)
st.download_button("📄 Download Registry Table CSV", df_detail.to_csv(index=False), "registry_table.csv", mime="text/csv")
st.download_button("📄 Download Registry Table JSON", json.dumps(detail_rows, indent=2, ensure_ascii=False), "registry_table.json", mime="application/json")

# 📑 Table Schema Preview
if not df_detail.empty:
    st.subheader("📑 Registry Table Schema")
    st.markdown(f"🧮 Columns: `{', '.join(df_detail.columns)}`")
    st.download_button("🧾 Download Schema JSON", json.dumps(list(df_detail.columns), indent=2, ensure_ascii=False), "registry_table_schema.json", mime="application/json")

# 💾 Export Layouts
st.header("📦 Export Box Layouts")
st.download_button("💾 Download Normalized Layouts", json.dumps(box_layouts, indent=2, ensure_ascii=False), "box_layouts_normalized.json", mime="application/json")

absolute_layouts = {
    zid: layout_managers[zid].load_layout(boxes)
    for zid, boxes in box_layouts.items()
}
st.download_button("💾 Download Absolute Layouts", json.dumps(absolute_layouts, indent=2, ensure_ascii=False), "box_layouts_absolute.json", mime="application/json")

# 📋 Layout Readiness & Extraction Audit
st.header("📊 Zone Extraction Audit")

for i, row in enumerate(metadata_rows, start=1):
    missing = [label for label in master_field_labels if label != "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ" and not row.get(label)]
    form_id = row.get("FormID", f"ZONE_{i}")
    st.markdown(f"🔹 Zone {i} — FormID: `{form_id}`")
    if missing:
        st.warning(f"⛔ Missing Master Fields: `{', '.join(missing)}`")
    else:
        st.success("✅ All required master fields present")

if detail_rows:
    st.markdown(f"📑 Total Detail Rows Extracted: `{len(detail_rows)}`")
else:
    st.warning("⚠️ No detail rows extracted — check Zone 3 layout or headers")

# 🟨 Highlight zones missing layout info
st.subheader("🟨 Missing Layout Zones")
for zid, manager in layout_managers.items():
    expected_labels = ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"] if zid == "2" else [l for l in master_field_labels if l != "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"]
    zone_boxes = box_layouts.get(zid, {})
    missing = [label for label in expected_labels if label not in zone_boxes or not all(v is not None for v in zone_boxes[label])]
    if missing:
        st.warning(f"⚠️ Zone {zid} is missing boxes for: `{', '.join(missing)}`")
    else:
        st.success(f"✅ Zone {zid} layout is complete")

# ==== END OF PART 5 ====
