# 📦 Core Libraries
import streamlit as st
import os, json, re, unicodedata
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from datetime import datetime
import pandas as pd
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

# 🔡 Normalize Greek Text from Latin & Cyrillic
def fix_latin_greek(text):
    return "".join({
        "A": "Α", "B": "Β", "E": "Ε", "H": "Η", "K": "Κ", "M": "Μ",
        "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ", "Y": "Υ"
    }.get(c, c) for c in text)

def fix_cyrillic_greek(text):
    return "".join({
        "А": "Α", "В": "Β", "С": "Σ", "Е": "Ε", "Н": "Η",
        "К": "Κ", "М": "Μ", "О": "Ο", "Р": "Ρ", "Т": "Τ", "Х": "Χ"
    }.get(c, c) for c in text)

def normalize(text):
    if not text: return ""
    text = fix_cyrillic_greek(fix_latin_greek(text))
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\sΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]", "", text)
    return text.upper().strip()

# 📅 Greek Date Normalization
def normalize_date(text):
    text = text.strip()
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y"]:
        try: return datetime.strptime(text, fmt).strftime("%d/%m/%Y")
        except: continue
    return text

# 🛡️ Field Validation
def validate_registry_field(label, corrected_text, confidence):
    issues = []
    greek_chars = re.findall(r"[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ]", corrected_text or "")
    if not corrected_text: issues.append("Missing")
    if label != "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ" and len(greek_chars) < max(3, len(corrected_text) // 2):
        issues.append("Non-Greek characters")
    if len(corrected_text) < 2: issues.append("Too short")
    if confidence < 50.0: issues.append("Low confidence")
    return issues

# 💡 Suggest Field Fix
def suggest_fix(label, corrected_text, issues):
    if "Too short" in issues or "Non-Greek characters" in issues:
        fixed = corrected_text.title()
        if len(fixed) >= 2 and re.match(r"^[Α-ΩΆΈΉΊΌΎΏ][α-ωάέήίόύώ]{2,}", fixed): return fixed
    return None

# ✂️ Image Trimming & Zone Splitting
def trim_whitespace(image, threshold=240, buffer=10):
    gray = image.convert("L")
    pixels = gray.load()
    w, h = image.size
    top = next((y for y in range(h) if any(pixels[x, y] < threshold for x in range(w))), 0)
    bottom = next((y for y in reversed(range(h)) if any(pixels[x, y] < threshold for x in range(w))), h)
    left = next((x for x in range(w) if any(pixels[x, y] < threshold for y in range(h))), 0)
    right = next((x for x in reversed(range(w)) if any(pixels[x, y] < threshold for y in range(h))), w)
    return image.crop((max(0,left-buffer), max(0,top-buffer), min(w,right+buffer), min(h,bottom+buffer)))

def crop_left(image):
    w,h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

def split_zones_fixed(image, overlap_px):
    w, h = image.size
    thirds = [int(h * t) for t in [0.0, 0.33, 0.66, 1.0]]
    bounds = [
        (thirds[0], thirds[1] + overlap_px),
        (thirds[1] - overlap_px, thirds[2] + overlap_px),
        (thirds[2] - overlap_px, thirds[3])
    ]
    zones = [image.crop((0, t, w, b)) for t, b in bounds]
    return zones, bounds

# 📐 Coordinate Conversion
def convert_box(box, image_size, to_normalized=True):
    if not box or any(v is None for v in box): return (None, None, None, None)
    x, y, w, h = box
    iw, ih = image_size
    return (
        (x / iw, y / ih, w / iw, h / ih) if to_normalized
        else (x * iw, y * ih, w * iw, h * ih)
    )

# 🧠 Confidence Estimator
def estimate_confidence(label, text):
    text = text.strip()
    if not text: return 0.0
    if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": return 90.0 if text.isdigit() else 40.0
    if label in ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]:
        return 75.0 if re.match(r"^[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ\s\-]{3,}$", text) else 30.0
    return 50.0

# 🩹 Vision OCR Fallback
def extract_field_from_box_with_vision(pil_img, box, label):
    try:
        x, y, w, h = convert_box(box, pil_img.size, to_normalized=False)
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cropped = pil_img.convert("RGB").crop((x1, y1, x2, y2))
        buf = BytesIO(); cropped.save(buf, format="JPEG")
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=buf.getvalue())
        response = client.text_detection(image=image, image_context={"language_hints": ["el"]})
        desc = response.text_annotations[0].description.strip() if response.text_annotations else ""
        return desc, estimate_confidence(label, desc)
    except Exception as e:
        st.warning(f"🛑 Vision OCR error for '{label}': {e}")
        return "", 0.0

# 🧠 Document AI Parser
def parse_docai(pil_img, project_id, processor_id, location):
    try:
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
        )
        name = client.processor_path(project_id, location, processor_id)
        buf = BytesIO(); pil_img.save(buf, format="JPEG")
        raw = documentai.RawDocument(content=buf.getvalue(), mime_type="image/jpeg")
        return client.process_document(request=documentai.ProcessRequest(name=name, raw_document=raw)).document
    except Exception as e:
        st.error(f"📛 Document AI error: {e}")
        return None

# 📄 Text Anchor Segment Resolver
def extract_text_from_anchor(anchor, full_text):
    if not anchor or not anchor.text_segments:
        return ""
    return "".join([
        full_text[int(seg.start_index):int(seg.end_index)]
        for seg in anchor.text_segments
        if seg.start_index is not None and seg.end_index is not None
    ]).strip()

# 🧭 LayoutManager Class
class LayoutManager:
    def __init__(self, image_size):
        self.image_size = image_size

    def to_pixel(self, box):
        return convert_box(box, self.image_size, to_normalized=False)

    def to_normalized(self, box):
        return convert_box(box, self.image_size, to_normalized=True)

    def load_layout(self, layout_dict):
        return {label: self.to_pixel(box) for label, box in layout_dict.items()}

    def save_layout(self, layout_dict):
        return {label: self.to_normalized(box) for label, box in layout_dict.items()}
# 🚀 Streamlit Config
st.set_page_config(page_title="📜 Greek Registry Parser", layout="wide")
st.title("📜 Greek Registry Parser — Metadata & Table Extraction")

# 📍 GCP Configuration
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

# 🧾 Registry Metadata Labels
target_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ",
    "ΕΠΩΝΥΜΟΝ",
    "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]

# 📦 App Containers
forms_parsed = []
manual_boxes_per_form = {}

# 🎛️ Sidebar Settings
st.sidebar.header("⚙️ Parser Settings")
overlap = st.sidebar.slider("🔁 Zone Overlap (px)", 0, 120, value=40)

# 🔐 GCP Credentials Loader
cred_file = st.sidebar.file_uploader("🔐 Upload GCP Credentials (.json)", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

# 📥 Optional Layout Import
uploaded_box_map = st.sidebar.file_uploader("📥 Load Box Layout Map (.json)", type=["json"])
if uploaded_box_map:
    try:
        manual_boxes_per_form = json.load(uploaded_box_map)
        st.sidebar.success(f"📦 Loaded layout for {len(manual_boxes_per_form)} forms")
    except Exception as e:
        st.sidebar.error(f"❌ Layout file error: {e}")

# 🖼️ Registry Scan Upload
uploaded_image = st.file_uploader("📄 Upload Registry Page", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("📎 Please upload a registry scan to begin.")
    st.stop()

# ✂️ Preprocessing
try:
    original = Image.open(uploaded_image)
    cropped = crop_left(trim_whitespace(original))
    zones, bounds = split_zones_fixed(cropped, overlap)
    if not zones:
        st.error("🚫 Zone detection failed.")
        st.stop()
except Exception as e:
    st.error(f"❌ Image preprocessing error: {e}")
    st.stop()

# 🧭 LayoutManagers Per Zone
layout_managers = {
    str(i + 1): LayoutManager(zones[i].size)
    for i in range(len(zones))
}

# 👀 Show Cropped Image & Zones
st.image(cropped, caption="📌 Cropped Registry Page (Left Side)", use_column_width=True)
st.header("🗂️ Zone Previews")
for i, zone in enumerate(zones, start=1):
    st.image(zone, caption=f"Zone {i}", width=280)
# 🔁 Loop over each zone
for idx, zone in enumerate(zones, start=1):
    fid = str(idx)
    manager = layout_managers[fid]
    zone_w, zone_h = zone.size

    st.header(f"📄 Form {fid}")

    # 🛠️ Fallback layout if missing
    if fid not in manual_boxes_per_form:
        manual_boxes_per_form[fid] = {
            "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": (0.05, 0.05, 0.15, 0.08),
            "ΕΠΩΝΥΜΟΝ":        (0.05, 0.15, 0.40, 0.07),
            "ΟΝΟΜΑ ΠΑΤΡΟΣ":    (0.05, 0.25, 0.40, 0.07),
            "ΟΝΟΜΑ ΜΗΤΡΟΣ":    (0.05, 0.35, 0.40, 0.07),
            "ΚΥΡΙΟΝ ΟΝΟΜΑ":    (0.05, 0.45, 0.40, 0.07)
        }
        st.warning("⚠️ Fallback layout applied")

    # 📝 Editable box layout
    layout_pixels = manager.load_layout(manual_boxes_per_form.get(fid, {}))
    editor_rows = []
    for label in target_labels:
        box = layout_pixels.get(label)
        x, y, w, h = box if box else (None, None, None, None)
        editor_rows.append({
            "Label": label,
            "X": x,
            "Y": y,
            "Width": w,
            "Height": h
        })

    editor_df = st.data_editor(
        pd.DataFrame(editor_rows),
        use_container_width=True,
        num_rows="dynamic",
        key=f"editor_{fid}"
    )

    # 💾 Save normalized layout
    edited_layout = {
        row["Label"]: (row["X"], row["Y"], row["Width"], row["Height"])
        for _, row in editor_df.iterrows()
        if all(v is not None for v in (row["X"], row["Y"], row["Width"], row["Height"]))
    }
    manual_boxes_per_form[fid] = manager.save_layout(edited_layout)

    # 🖼️ Overlay preview
    overlay = zone.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype("arial.ttf", size=14) if os.path.exists("arial.ttf") else None

    for label, box in manual_boxes_per_form[fid].items():
        try:
            x, y, w, h = manager.to_pixel(box)
            if None in (x, y, w, h): continue
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            is_valid = x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and x2 <= zone_w and y2 <= zone_h
            color = "purple" if is_valid else "red"
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            draw.text((x1, y1 - 16), f"{label}{' ⚠️' if not is_valid else ''}", fill=color, font=font)
        except Exception as e:
            st.warning(f"❌ Overlay error for '{label}': {e}")

    st.image(overlay, caption="📌 Metadata Field Boxes", use_container_width=True)

    # 🔍 Document AI Extraction
    doc = parse_docai(zone.copy(), project_id, processor_id, location)
    extracted_fields = []
    if doc:
        for page in doc.pages:
            for f in page.form_fields:
                label_raw = extract_text_from_anchor(f.field_name.text_anchor, doc.text)
                value_raw = extract_text_from_anchor(f.field_value.text_anchor, doc.text)
                conf = round(f.field_value.confidence * 100, 2)
                for target in target_labels:
                    if normalize(label_raw) == normalize(target):
                        corrected = normalize(value_raw)
                        issues = validate_registry_field(target, corrected, conf)
                        suggestion = suggest_fix(target, corrected, issues)
                        extracted_fields.append({
                            "Label": target,
                            "Raw": value_raw,
                            "Corrected": corrected,
                            "Confidence": conf,
                            "Issues": issues,
                            "Suggestion": suggestion,
                            "Thumb": None
                        })

    # 🩹 Fill missing fields
    for label in target_labels:
        exists = any(f["Label"] == label for f in extracted_fields)
        if exists: continue

        box = manual_boxes_per_form[fid].get(label)
        text, conf = extract_field_from_box_with_vision(zone, box, label) if box else ("", 0.0)
        corrected = normalize(text)
        issues = validate_registry_field(label, corrected, conf)
        suggestion = suggest_fix(label, corrected, issues)

        thumb = None
        if box:
            try:
                x, y, w, h = manager.to_pixel(box)
                thumb = zone.crop((int(x), int(y), int(x + w), int(y + h))).convert("RGB")
            except: pass

        extracted_fields.append({
            "Label": label,
            "Raw": text,
            "Corrected": corrected,
            "Confidence": conf,
            "Issues": issues,
            "Suggestion": suggestion,
            "Thumb": thumb
        })

    # 📊 Table Parsing — First 6 Columns Only
    table_rows = []
    if doc:
        for page in doc.pages:
            for table in page.tables:
                headers = []
                for header_row in table.header_rows:
                    for cell in header_row.cells[:6]:
                        raw = extract_text_from_anchor(cell.layout.text_anchor, doc.text)
                        headers.append(normalize(raw) or f"COL_{len(headers)}")

                st.markdown(f"🧾 Table Schema (Form {fid}): `{', '.join(headers)}`")
                st.markdown(f"🧮 Columns Parsed: **{len(headers)}**")

                for row in table.body_rows:
                    data = {}
                    for i in range(min(6, len(row.cells))):
                        cell = row.cells[i]
                        key = headers[i] if i < len(headers) else f"COL_{i}"
                        value = extract_text_from_anchor(cell.layout.text_anchor, doc.text)
                        if "ΗΜΕΡ" in key:
                            value = normalize_date(value)
                        data[key] = normalize(value)
                    table_rows.append(data)

    # 💾 Save Form Data
    forms_parsed.append({
        "Form": fid,
        "Fields": extracted_fields,
        "TableRows": table_rows
    })
# 🧠 Final Review Dashboard
st.header("📊 Metadata & Table Review")
apply_all = st.checkbox("💡 Auto-apply Suggested Fixes", value=False)
flat_fields = []

# 🔁 Loop through parsed forms
for form in forms_parsed:
    fid = form["Form"]
    fields = form["Fields"]

    st.subheader(f"📄 Form {fid}")
    col1, col2 = st.columns(2)

    # 🔍 Field Summary
    with col1:
        df_summary = pd.DataFrame([
            {
                "Label": f["Label"],
                "Raw": f["Raw"],
                "Corrected": f["Corrected"],
                "Confidence": f["Confidence"],
                "Issues": ", ".join(f["Issues"])
            } for f in fields
        ])
        st.markdown("🔍 Metadata Fields")
        st.dataframe(df_summary, use_container_width=True)

    # ✏️ Final Corrections
    with col2:
        st.markdown("✏️ Review & Override")
        for f in fields:
            suggestion = f.get("Suggestion")
            corrected = f["Corrected"]
            default = suggestion if apply_all and suggestion else corrected
            f["Final"] = st.text_input(
                f"{f['Label']} (Suggested: {suggestion or '—'})",
                value=default,
                key=f"final_{fid}_{f['Label']}"
            )

            thumb = f.get("Thumb")
            if thumb and f["Issues"]:
                st.image(thumb.convert("RGB"), caption=f"{f['Label']} ⚠️ {', '.join(f['Issues'])}", width=240)

    # 📥 Add flat field rows
    flat_fields.extend([
        {
            "Form": fid,
            "Label": f["Label"],
            "Raw": f["Raw"],
            "Corrected": f["Corrected"],
            "Final": f["Final"],
            "Confidence": f["Confidence"],
            "Issues": f["Issues"],
            "Suggestion": f["Suggestion"],
            "Thumb": "Yes" if f.get("Thumb") else "No"
        } for f in fields
    ])

    # 🗃️ Display table rows
    table = form.get("TableRows", [])
    if table:
        st.subheader("🗃️ Registry Table")
        df_table = pd.DataFrame(table)
        st.dataframe(df_table, use_container_width=True)

        st.download_button(
            label=f"📥 Download Table CSV — Form {fid}",
            data=df_table.to_csv(index=False),
            file_name=f"registry_table_form_{fid}.csv",
            mime="text/csv"
        )

        # 📤 Schema Export
        schema_headers = list(df_table.columns)
        schema_json = json.dumps(schema_headers, indent=2, ensure_ascii=False)
        st.download_button(
            label=f"🧾 Download Table Schema — Form {fid}",
            data=schema_json,
            file_name=f"registry_schema_form_{fid}.json",
            mime="application/json"
        )

# 📤 Final Metadata Export
st.header("📤 Export Metadata Fields")
df_export = pd.DataFrame(flat_fields)

st.download_button(
    label="📄 Download Metadata CSV",
    data=df_export.drop(columns=["Thumb"]).to_csv(index=False),
    file_name="registry_metadata_fields.csv",
    mime="text/csv"
)

st.download_button(
    label="📄 Download Metadata JSON",
    data=json.dumps(flat_fields, indent=2, ensure_ascii=False),
    file_name="registry_metadata_fields.json",
    mime="application/json"
)

# 📈 Confidence Summary
st.subheader("📈 Confidence Summary")
if not df_export.empty:
    avg_conf = round(df_export["Confidence"].mean(), 2)
    st.markdown(f"📌 Average Confidence: **{avg_conf}%**")

    low_conf = df_export[df_export["Confidence"] < 50.0]
    if not low_conf.empty:
        st.markdown("🔍 Fields < 50% Confidence")
        st.dataframe(low_conf.drop(columns=["Thumb"]), use_container_width=True)
else:
    st.info("ℹ️ No field data available.")

# 🚨 Validation Warnings
st.subheader("🚨 Validation Issues")
problem_fields = [f for f in flat_fields if f["Issues"]]
if problem_fields:
    df_problems = pd.DataFrame(problem_fields)
    st.dataframe(df_problems.drop(columns=["Thumb"]), use_container_width=True)
else:
    st.markdown("✅ No issues detected.")

# 💡 Suggestions Recap
st.subheader("💡 Applied Suggestions")
used_suggestions = [f for f in flat_fields if f["Suggestion"]]
if used_suggestions:
    for f in used_suggestions:
        st.markdown(f"**Form {f['Form']} — {f['Label']}**")
        st.markdown(f"🔍 `{f['Corrected']}` → Suggested: `{f['Suggestion']}` → Final: `{f['Final']}`")
        st.markdown("---")
else:
    st.markdown("🟢 No suggestions applied.")

# 📦 Layout Map Exports
st.header("📦 Export Layout Maps")

st.download_button(
    label="💾 Download Normalized Layouts (JSON)",
    data=json.dumps(manual_boxes_per_form, indent=2, ensure_ascii=False),
    file_name="layout_normalized.json",
    mime="application/json"
)

export_absolute = {
    fid: layout_managers[fid].load_layout(layout)
    for fid, layout in manual_boxes_per_form.items()
}
st.download_button(
    label="💾 Download Absolute Layouts (JSON)",
    data=json.dumps(export_absolute, indent=2, ensure_ascii=False),
    file_name="layout_absolute.json",
    mime="application/json"
)
