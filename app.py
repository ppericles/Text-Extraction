# 📦 Core Imports
import streamlit as st
import os, json, re, unicodedata
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from datetime import datetime
import pandas as pd
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

# 🔠 Normalize Latin & Cyrillic → Greek
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

# 🧼 Normalize Text for Registry Use
def normalize(text):
    if not text: return ""
    text = fix_cyrillic_greek(fix_latin_greek(text))
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\sΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]", "", text)
    return text.upper().strip()

# 📅 Normalize Greek-style Date
def normalize_date(text):
    text = text.strip()
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y"]:
        try: return datetime.strptime(text, fmt).strftime("%d/%m/%Y")
        except: continue
    return text

# 🛡️ Registry Field Validation
def validate_registry_field(label, corrected_text, confidence):
    issues = []
    greek_chars = re.findall(r"[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ]", corrected_text or "")
    if not corrected_text: issues.append("Missing")
    if label != "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ" and len(greek_chars) < max(3, len(corrected_text)//2): issues.append("Non-Greek characters")
    if len(corrected_text) < 2: issues.append("Too short")
    if confidence < 50.0: issues.append("Low confidence")
    return issues

# 💡 Suggest Fix for Problematic Values
def suggest_fix(label, corrected_text, issues):
    if "Too short" in issues or "Non-Greek characters" in issues:
        fixed = corrected_text.title()
        if len(fixed) >= 2 and re.match(r"^[Α-ΩΆΈΉΊΌΎΏ][α-ωάέήίόύώ]{2,}", fixed): return fixed
    return None

# ✂️ Whitespace Cropper
def trim_whitespace(image, threshold=240, buffer=10):
    gray = image.convert("L")
    pixels = gray.load()
    w, h = image.size
    top = next((y for y in range(h) if any(pixels[x,y] < threshold for x in range(w))),0)
    bottom = next((y for y in reversed(range(h)) if any(pixels[x,y] < threshold for x in range(w))),h)
    left = next((x for x in range(w) if any(pixels[x,y] < threshold for y in range(h))),0)
    right = next((x for x in reversed(range(w)) if any(pixels[x,y] < threshold for y in range(h))),w)
    return image.crop((max(0,left-buffer),max(0,top-buffer),min(w,right+buffer),min(h,bottom+buffer)))

# ✂️ Crop Left Side of Registry Image
def crop_left(image):
    w,h = image.size
    return image.convert("RGB").crop((0,0,w//2,h))

# 🔁 Fixed Vertical Zone Splitter (with Overlap)
def split_zones_fixed(image, overlap_px):
    w,h = image.size
    thirds = [int(h*t) for t in [0.0,0.33,0.66,1.0]]
    bounds = [
        (thirds[0], thirds[1]+overlap_px),
        (thirds[1]-overlap_px, thirds[2]+overlap_px),
        (thirds[2]-overlap_px, thirds[3])
    ]
    zones = [image.crop((0,t,w,b)) for t,b in bounds]
    return zones, bounds

# 📐 Convert Box Coordinates Normalized ↔ Absolute
def convert_box(box, image_size, to_normalized=True):
    if not box or any(v is None for v in box): return (None,None,None,None)
    x,y,w,h = box
    iw,ih = image_size
    return (
        (x/iw, y/ih, w/iw, h/ih) if to_normalized
        else (x*iw, y*ih, w*iw, h*ih)
    )

# 🧠 Estimate Confidence of OCR Result
def estimate_confidence(label, text):
    text = text.strip()
    if not text: return 0.0
    if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": return 90.0 if text.isdigit() else 40.0
    if label in ["ΕΠΩΝΥΜΟΝ","ΟΝΟΜΑ ΠΑΤΡΟΣ","ΟΝΟΜΑ ΜΗΤΡΟΣ","ΚΥΡΙΟΝ ΟΝΟΜΑ"]:
        return 75.0 if re.match(r"^[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ\s\-]{3,}$", text) else 30.0
    return 50.0

# 🩹 Vision API OCR Fallback
def extract_field_from_box_with_vision(pil_img, box, label):
    try:
        x, y, w, h = convert_box(box, pil_img.size, to_normalized=False)
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
        cropped = pil_img.convert("RGB").crop((x1, y1, x2, y2))
        buf = BytesIO(); cropped.save(buf, format="JPEG")
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=buf.getvalue())
        response = client.text_detection(image=image, image_context={"language_hints":["el"]})
        desc = response.text_annotations[0].description.strip() if response.text_annotations else ""
        return desc, estimate_confidence(label, desc)
    except Exception as e:
        st.warning(f"🛑 Vision OCR error for '{label}': {e}")
        return "", 0.0

# 🧠 Document AI Parser Wrapper
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

# 🧭 LayoutManager for Reliable Box Scaling
class LayoutManager:
    def __init__(self, image_size):
        self.image_size = image_size  # (width, height)

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
st.title("📜 Greek Registry Parser — Document AI + Vision OCR")

# 📍 GCP Configuration
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

# 🧾 Registry Labels to Parse
target_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ",
    "ΕΠΩΝΥΜΟΝ",
    "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]

# 📦 Runtime Containers
forms_parsed = []
manual_boxes_per_form = {}

# 🎛️ Sidebar Controls
st.sidebar.header("⚙️ Parser Settings")
overlap = st.sidebar.slider("🔁 Zone Overlap (pixels)", 0, 120, value=40)
normalize_input = st.sidebar.checkbox("📏 Normalize Box Inputs", value=True)

# 🔐 GCP Credentials Upload
cred_file = st.sidebar.file_uploader("🔐 GCP Credentials (JSON)", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

# 📥 Optional Box Layout Map
uploaded_box_map = st.sidebar.file_uploader("📥 Load Box Layout (JSON)", type=["json"])
if uploaded_box_map:
    try:
        manual_boxes_per_form = json.load(uploaded_box_map)
        st.sidebar.success(f"📦 Loaded layout for {len(manual_boxes_per_form)} forms")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to parse layout map: {e}")

# 🖼️ Registry Scan Upload
uploaded_image = st.file_uploader("📄 Upload Registry Scan (JPG, PNG)", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("📎 Please upload a registry image to begin.")
    st.stop()

# ✂️ Image Preprocessing
try:
    original = Image.open(uploaded_image)
    cropped = crop_left(trim_whitespace(original))
    zones, bounds = split_zones_fixed(cropped, overlap)
    if not zones:
        st.error("🚫 No zones detected in image.")
        st.stop()
except Exception as e:
    st.error(f"🛑 Image processing failed: {e}")
    st.stop()

# 🧭 LayoutManagers for Scaling per Zone
layout_managers = {
    str(i + 1): LayoutManager(zones[i].size)
    for i in range(len(zones))
}

# 👀 Show Cropped Image and Zone Previews
st.image(cropped, caption="📌 Cropped Registry Image (Left Side)", use_column_width=True)
st.header("🗂️ Zone Previews")
for i, zone in enumerate(zones, start=1):
    st.image(zone, caption=f"Zone {i}", width=280)
# 🔁 Loop through zones
for idx, zone in enumerate(zones, start=1):
    fid = str(idx)
    manager = layout_managers[fid]
    zone_w, zone_h = zone.size

    st.header(f"📄 Form {fid}")

    # 🛠️ Auto-generate fallback layout if missing
    if fid not in manual_boxes_per_form:
        aspect_ratio = round(zone_w / zone_h, 2)
        if 0.7 <= aspect_ratio <= 1.6 and zone_h > 600:
            fallback_layout = {
                "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": (0.05, 0.04, 0.15, 0.08),
                "ΕΠΩΝΥΜΟΝ":        (0.05, 0.15, 0.40, 0.07),
                "ΟΝΟΜΑ ΠΑΤΡΟΣ":    (0.05, 0.25, 0.40, 0.07),
                "ΟΝΟΜΑ ΜΗΤΡΟΣ":    (0.05, 0.35, 0.40, 0.07),
                "ΚΥΡΙΟΝ ΟΝΟΜΑ":    (0.05, 0.45, 0.40, 0.07)
            }
            manual_boxes_per_form[fid] = fallback_layout
            st.warning(f"⚠️ No layout found — fallback applied for Form {fid}")

    # 📝 Build editable box editor
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

    # 💾 Save layout as normalized
    edited_layout = {
        row["Label"]: (row["X"], row["Y"], row["Width"], row["Height"])
        for _, row in editor_df.iterrows()
        if all(val is not None for val in (row["X"], row["Y"], row["Width"], row["Height"]))
    }
    manual_boxes_per_form[fid] = manager.save_layout(edited_layout)

    # 🖼️ Overlay display with invalid boxes in red
    overlay = zone.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype("arial.ttf", size=14) if os.path.exists("arial.ttf") else None

    for label, box in manual_boxes_per_form[fid].items():
        try:
            x, y, w, h = manager.to_pixel(box)
            if None in (x, y, w, h):
                st.warning(f"⚠️ Box for '{label}' is incomplete.")
                continue
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            is_valid = (
                x1 >= 0 and y1 >= 0 and
                x2 > x1 and y2 > y1 and
                x2 <= zone_w and y2 <= zone_h
            )
            color = "purple" if is_valid else "red"
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            draw.text((x1, y1 - 16), f"{label}{' ⚠️' if not is_valid else ''}", fill=color, font=font)

        except Exception as e:
            st.warning(f"❌ Overlay error for '{label}': {e}")

    st.image(overlay, caption="📌 Annotated Boxes (Red = Invalid)", use_container_width=True)

    # 🔍 Document AI field extraction
    doc = parse_docai(zone.copy(), project_id, processor_id, location)
    extracted = {}
    if doc:
        for page in doc.pages:
            for f in page.form_fields:
                label_raw = f.field_name.text_anchor.content or ""
                value_raw = f.field_value.text_anchor.content or ""
                conf = round(f.field_value.confidence * 100, 2)
                for target in target_labels:
                    if normalize(label_raw) == normalize(target):
                        corrected = normalize(value_raw)
                        issues = validate_registry_field(target, corrected, conf)
                        suggestion = suggest_fix(target, corrected, issues)
                        extracted[target] = {
                            "Label": target,
                            "Raw": value_raw.strip(),
                            "Corrected": corrected,
                            "Confidence": conf,
                            "Issues": issues,
                            "Suggestion": suggestion,
                            "Thumb": None
                        }

    # 🩹 Fill missing fields via Vision OCR
    fields = []
    for label in target_labels:
        f = extracted.get(label)
        if f and f["Raw"]:
            fields.append(f)
            continue

        box = manual_boxes_per_form[fid].get(label)
        text, conf = extract_field_from_box_with_vision(zone, box, label) if box else ("", 0.0)
        corrected = normalize(text)
        issues = validate_registry_field(label, corrected, conf)
        suggestion = suggest_fix(label, corrected, issues)

        thumb = None
        if box:
            try:
                x, y, w, h = manager.to_pixel(box)
                thumb = zone.crop((int(x), int(y), int(x+w), int(y+h))).convert("RGB")
            except Exception as e:
                st.warning(f"⚠️ Couldn't extract thumbnail for '{label}': {e}")

        fields.append({
            "Label": label,
            "Raw": text,
            "Corrected": corrected,
            "Confidence": conf,
            "Issues": issues,
            "Suggestion": suggestion,
            "Thumb": thumb
        })

    # 📊 Parse registry tables (if present)
    table_rows = []
    if doc:
        for page in doc.pages:
            for table in page.tables:
                headers = []
                for header_row in table.header_rows:
                    headers = [
                        cell.layout.text_anchor.content.strip()
                        if cell.layout.text_anchor.content else ""
                        for cell in header_row.cells
                    ]
                key_map = {
                    "Α/Α": "Index",
                    "ΤΟΜΟΣ": "Volume",
                    "ΑΡΙΘ.": "RegistryNumber",
                    "ΗΜΕΡΟΜΗΝΙΑ ΜΕΤΑΓΡΑΦΗΣ": "TransferDate",
                    "ΑΡΙΘ. ΕΓΓΡΑΦΟΥ ΚΑΙ ΕΤΟΣ ΑΥΤΟΥ": "DocumentNumber",
                    "ΣΥΜΒΟΛΑΙΟΓΡΑΦΟΣ Ή Η ΕΚΔΟΥΣΑ ΑΡΧΗ": "Issuer"
                }
                for body_row in table.body_rows:
                    row_data = {}
                    for i, cell in enumerate(body_row.cells):
                        header = headers[i] if i < len(headers) else f"COL_{i}"
                        key = key_map.get(normalize(header), header)
                        value = cell.layout.text_anchor.content.strip() if cell.layout.text_anchor.content else ""
                        if key == "TransferDate":
                            value = normalize_date(value)
                        row_data[key] = normalize(value)
                    table_rows.append(row_data)

    # 💾 Save parsed data
    forms_parsed.append({
        "Form": fid,
        "Fields": fields,
        "TableRows": table_rows,
        "Missing": [f["Label"] for f in fields if not f["Raw"].strip()]
    })
# 🧠 Final Review Panel
st.header("📊 Registry Review")

apply_all = st.checkbox("💡 Auto-apply suggested fixes", value=False)
flat_fields = []

# 🔁 Loop through parsed forms
for form in forms_parsed:
    fid = form["Form"]
    fields = form["Fields"]

    st.subheader(f"📄 Form {fid}")
    col1, col2 = st.columns(2)

    # 🔍 Summary Table
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
        st.markdown("🔍 Parsed Fields")
        st.dataframe(df_summary, use_container_width=True)

    # ✏️ Manual Review
    with col2:
        st.markdown("✏️ Final Corrections")
        for f in fields:
            suggestion = f.get("Suggestion")
            corrected = f["Corrected"]
            default = suggestion if apply_all and suggestion else corrected
            f["Final"] = st.text_input(
                f"{f['Label']} (Suggested: {suggestion or '—'})",
                value=default,
                key=f"final_{fid}_{f['Label']}"
            )

            # 🖼️ Show thumbnail if issues flagged
            thumb = f.get("Thumb")
            if thumb and f["Issues"]:
                st.image(thumb.convert("RGB"), caption=f"{f['Label']} ⚠️ {', '.join(f['Issues'])}", width=240)

    # 📥 Export flat field rows
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

    # 📄 Registry Table Export
    if form.get("TableRows"):
        st.subheader("🗃️ Registry Table Rows")
        df_table = pd.DataFrame(form["TableRows"])
        st.dataframe(df_table, use_container_width=True)

        st.download_button(
            label=f"📥 Download Registry Table for Form {fid}",
            data=df_table.to_csv(index=False),
            file_name=f"registry_table_form_{fid}.csv",
            mime="text/csv"
        )

# 📤 Export Final Registry Fields
st.header("📤 Export Parsed Fields")

df_export = pd.DataFrame(flat_fields)

st.download_button(
    label="📄 Download Fields (CSV)",
    data=df_export.drop(columns=["Thumb"]).to_csv(index=False),
    file_name="registry_fields.csv",
    mime="text/csv"
)

st.download_button(
    label="📄 Download Fields (JSON)",
    data=json.dumps(flat_fields, indent=2, ensure_ascii=False),
    file_name="registry_fields.json",
    mime="application/json"
)

# 📈 Confidence Summary
st.subheader("📈 Confidence Report")
if not df_export.empty:
    avg_conf = round(df_export["Confidence"].mean(), 2)
    st.markdown(f"📌 Average Confidence: **{avg_conf}%**")

    low_conf_df = df_export[df_export["Confidence"] < 50.0]
    if not low_conf_df.empty:
        st.markdown("🔍 Low Confidence Fields (< 50%)")
        st.dataframe(low_conf_df.drop(columns=["Thumb"]), use_container_width=True)
else:
    st.info("ℹ️ No field data available for confidence summary.")

# 🚨 Validation Alerts
st.subheader("🚨 Validation Issues")
problem_fields = [f for f in flat_fields if f["Issues"]]
if problem_fields:
    df_problems = pd.DataFrame(problem_fields)
    st.dataframe(df_problems.drop(columns=["Thumb"]), use_container_width=True)
else:
    st.markdown("✅ No flagged issues found.")

# 💡 Suggestions Recap
st.subheader("💡 Suggestions Applied")
used_suggestions = [f for f in flat_fields if f["Suggestion"]]
if used_suggestions:
    for f in used_suggestions:
        st.markdown(f"**Form {f['Form']} — {f['Label']}**")
        st.markdown(f"🔍 Parsed: `{f['Corrected']}` → Suggested: `{f['Suggestion']}` → Final: `{f['Final']}`")
        st.markdown("---")
else:
    st.markdown("🟢 No suggestions applied.")

# 📦 Export Layout Maps
st.header("📦 Layout Map Exports")

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
