import streamlit as st
import os, json, re, unicodedata
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from datetime import datetime
import pandas as pd
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

# 🔠 Latin-to-Greek character fixer
def fix_latin_greek(text):
    return "".join({
        "A": "Α", "B": "Β", "E": "Ε", "H": "Η", "K": "Κ", "M": "Μ",
        "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ", "Y": "Υ"
    }.get(c, c) for c in text)

# 🔠 Cyrillic-to-Greek character fixer
def fix_cyrillic_greek(text):
    return "".join({
        "А": "Α", "В": "Β", "С": "Σ", "Е": "Ε", "Н": "Η",
        "К": "Κ", "М": "Μ", "О": "Ο", "Р": "Ρ", "Т": "Τ", "Х": "Χ"
    }.get(c, c) for c in text)

# 🧹 Normalize and sanitize Greek registry text
def normalize(text):
    if not text:
        return ""
    text = fix_cyrillic_greek(fix_latin_greek(text))
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\sΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]", "", text)
    return text.upper().strip()

# 🗓️ Date parser to DD/MM/YYYY format
def normalize_date(text):
    text = text.strip()
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y"]:
        try:
            return datetime.strptime(text, fmt).strftime("%d/%m/%Y")
        except:
            continue
    return text

# 🧠 Field validator
def validate_registry_field(label, corrected_text, confidence):
    issues = []
    is_numeric = label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"
    greek_chars = re.findall(r"[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ]", corrected_text or "")
    if not corrected_text:
        issues.append("Missing")
    if not is_numeric and len(greek_chars) < max(3, len(corrected_text) // 2):
        issues.append("Non-Greek characters")
    if len(corrected_text) < 2:
        issues.append("Too short")
    if confidence < 50.0:
        issues.append("Low confidence")
    return issues

# 💡 Fix suggester
def suggest_fix(label, corrected_text, issues):
    if "Too short" in issues or "Non-Greek characters" in issues:
        fixed = corrected_text.title()
        if len(fixed) >= 2 and re.match(r"^[Α-ΩΆΈΉΊΌΎΏ][α-ωάέήίόύώ]{2,}", fixed):
            return fixed
    return None

# ✂️ Whitespace cropper
def trim_whitespace(image, threshold=240, buffer=10):
    gray = image.convert("L")
    pixels = gray.load()
    w, h = image.size
    top = next((y for y in range(h) if any(pixels[x, y] < threshold for x in range(w))), 0)
    bottom = next((y for y in reversed(range(h)) if any(pixels[x, y] < threshold for x in range(w))), h)
    left = next((x for x in range(w) if any(pixels[x, y] < threshold for y in range(h))), 0)
    right = next((x for x in reversed(range(w)) if any(pixels[x, y] < threshold for y in range(h))), w)
    return image.crop((max(0, left - buffer), max(0, top - buffer), min(w, right + buffer), min(h, bottom + buffer)))

# ✂️ Left-half cropper
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# ✂️ Fixed zone splitter (with overlap)
def split_zones_fixed(image, overlap_px):
    w, h = image.size
    thirds = [int(h * t) for t in [0.0, 0.33, 0.66, 1.0]]
    bounds = [(thirds[0], thirds[1] + overlap_px), (thirds[1] - overlap_px, thirds[2] + overlap_px), (thirds[2] - overlap_px, thirds[3])]
    return [image.crop((0, t, w, b)) for t, b in bounds], bounds

# 🧠 Estimate field confidence
def estimate_confidence(label, text):
    text = text.strip()
    if not text:
        return 0.0
    if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ":
        return 90.0 if text.isdigit() else 40.0
    elif label in ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]:
        return 75.0 if re.match(r"^[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ\s\-]{3,}$", text) else 30.0
    return 50.0

# 🩹 Vision OCR fallback with crop safety
def extract_field_from_box_with_vision(pil_img, box, label):
    try:
        x, y, bw, bh = [float(v) for v in box]
        w, h = pil_img.size
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + bw) * w)
        y2 = int((y + bh) * h)
        cropped = pil_img.convert("RGB").crop((x1, y1, x2, y2))
    except Exception as e:
        st.warning(f"⚠️ Crop error for {label}: {e}")
        return "", 0.0

    try:
        buf = BytesIO()
        cropped.save(buf, format="JPEG")
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=buf.getvalue())
        context = {"language_hints": ["el"]}
        response = client.text_detection(image=image, image_context=context)
        if response.error.message:
            st.warning(f"🛑 Vision error for {label}: {response.error.message}")
            return "", 0.0
        desc = response.text_annotations[0].description.strip() if response.text_annotations else ""
        return desc, estimate_confidence(label, desc)
    except Exception as e:
        st.warning(f"🛑 OCR failed for {label}: {e}")
        return "", 0.0

# 🧠 Document AI wrapper
def parse_docai(pil_img, project_id, processor_id, location):
    try:
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
        )
        name = client.processor_path(project_id, location, processor_id)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        raw = documentai.RawDocument(content=buf.getvalue(), mime_type="image/jpeg")
        response = client.process_document(request=documentai.ProcessRequest(name=name, raw_document=raw))
        return response.document
    except Exception as e:
        st.error(f"📛 Document AI error: {e}")
        return None
# 🚀 Streamlit page setup
st.set_page_config(page_title="📜 Greek Registry Parser", layout="wide")
st.title("📜 Greek Registry Parser — Document AI + Vision Review")

# 📦 Global registry settings
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

# 🧰 Sidebar configuration
st.sidebar.header("⚙️ Parser Settings")

overlap = st.sidebar.slider("🔁 Zone Overlap (px)", 0, 120, value=40)
normalize_input = st.sidebar.checkbox("📏 Normalize Box Inputs", value=True)

cred_file = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

uploaded_box_map = st.sidebar.file_uploader("📥 Import Fallback Box Map", type=["json"])
manual_boxes_per_form = {}

if uploaded_box_map:
    try:
        manual_boxes_per_form = json.load(uploaded_box_map)
        st.sidebar.success(f"✅ Loaded box layout for {len(manual_boxes_per_form)} form(s)")
    except Exception as e:
        st.sidebar.error(f"❌ Could not load box map: {e}")

# 🖼️ Upload registry image
uploaded_image = st.file_uploader("🖼️ Upload Registry Image", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("ℹ️ Please upload a registry page to begin.")
    st.stop()

# ✂️ Trim whitespace and crop left half
try:
    original = Image.open(uploaded_image)
    cropped = crop_left(trim_whitespace(original))
    if cropped.size == (0, 0):
        st.error("❌ Cropped image is blank — check scan resolution.")
        st.stop()
    zones, bounds = split_zones_fixed(cropped, overlap)
    if not zones:
        st.error("❌ Failed to split registry image into vertical zones.")
        st.stop()
except Exception as e:
    st.error(f"🛑 Image preprocessing failed: {e}")
    st.stop()

# 🖼️ Display previews of cropped and split zones
st.image(cropped, caption="📎 Cropped Registry (Left Half)", use_container_width=True)
st.header("🧾 Preview of Registry Zones")

for i, zone in enumerate(zones, start=1):
    st.image(zone, caption=f"Zone {i}", width=300)
# 🔠 Registry Fields to Extract
target_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ",
    "ΕΠΩΝΥΜΟΝ",
    "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]

# 🗂️ Store all parsed zone data
forms_parsed = []

# 🔁 Loop through each zone
for idx, zone in enumerate(zones, start=1):
    st.header(f"📄 Form {idx}")
    zone_w, zone_h = zone.size

    # 📐 Editable fallback box setup
    initial_boxes = manual_boxes_per_form.get(str(idx), {})
    editor_rows = []

    for label in target_labels:
        box = initial_boxes.get(label, (None, None, None, None))
        try:
            x, y, w, h = [float(v) for v in box]
            if normalize_input:
                x /= zone_w
                y /= zone_h
                w /= zone_w
                h /= zone_h
        except:
            x, y, w, h = None, None, None, None
        editor_rows.append({"Label": label, "X": x, "Y": y, "Width": w, "Height": h})

    editor_df = st.data_editor(
        pd.DataFrame(editor_rows),
        use_container_width=True,
        num_rows="dynamic",
        key=f"box_editor_{idx}"
    )

    manual_boxes_per_form[str(idx)] = {
        row["Label"]: (row["X"], row["Y"], row["Width"], row["Height"])
        for _, row in editor_df.iterrows()
        if all(val is not None for val in (row["X"], row["Y"], row["Width"], row["Height"]))
    }

    # ↔️ Propagate Form 1 layout to all remaining forms
    if idx == 1:
        layout_1 = manual_boxes_per_form["1"]
        for z in range(2, len(zones) + 1):
            fid = str(z)
            if fid not in manual_boxes_per_form:
                manual_boxes_per_form[fid] = layout_1
        st.info("📐 Applied Form 1 layout to all zones")

    # 🎨 Draw overlay of boxes
    overlay = zone.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype("arial.ttf", size=14) if os.path.exists("arial.ttf") else None
    for label, box in manual_boxes_per_form[str(idx)].items():
        try:
            x, y, bw, bh = [float(v) for v in box]
            w, h = overlay.size
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + bw) * w)
            y2 = int((y + bh) * h)
            draw.rectangle([(x1, y1), (x2, y2)], outline="purple", width=2)
            draw.text((x1, y1 - 16), label, fill="purple", font=font)
        except Exception as e:
            st.warning(f"⚠️ Overlay error for '{label}': {e}")
    st.image(overlay, caption=f"🟣 Annotated Boxes — Form {idx}", use_container_width=True)

    # 🔍 Run Document AI parsing
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
                        corrected = normalize(fix_cyrillic_greek(fix_latin_greek(value_raw)))
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

    # 🩹 Vision fallback if missing
    fields = []
    for label in target_labels:
        f = extracted.get(label)
        if f and f["Raw"]:
            fields.append(f)
            continue

        box = manual_boxes_per_form[str(idx)].get(label)
        text, conf = extract_field_from_box_with_vision(zone, box, label) if box else ("", 0.0)
        corrected = normalize(fix_cyrillic_greek(fix_latin_greek(text)))
        issues = validate_registry_field(label, corrected, conf)
        suggestion = suggest_fix(label, corrected, issues)

        thumb = None
        try:
            x, y, bw, bh = [float(v) for v in box]
            w, h = zone.size
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + bw) * w)
            y2 = int((y + bh) * h)
            thumb = zone.crop((x1, y1, x2, y2)).convert("RGB")
        except:
            thumb = None

        fields.append({
            "Label": label,
            "Raw": text,
            "Corrected": corrected,
            "Confidence": conf,
            "Issues": issues,
            "Suggestion": suggestion,
            "Thumb": thumb
        })

    # 🗃️ Extract 6-column table rows
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

    # 📋 Store results
    forms_parsed.append({
        "Form": idx,
        "Fields": fields,
        "TableRows": table_rows,
        "Missing": [f["Label"] for f in fields if not f["Raw"].strip()]
    })
# 🧠 Review Dashboard
st.header("📊 Registry Review and Export Dashboard")

apply_all = st.checkbox("🧠 Apply Suggested Corrections Automatically", value=False)
flat_fields = []

for form in forms_parsed:
    form_id = form["Form"]
    fields = form["Fields"]

    st.subheader(f"📄 Form {form_id}")
    col1, col2 = st.columns(2)

    # 🔍 Left Column: Parsed Field Info
    with col1:
        st.markdown("🔍 Parsed Fields")
        parsed_df = pd.DataFrame([
            {
                "Label": f["Label"],
                "Raw": f["Raw"],
                "Corrected": f["Corrected"],
                "Confidence": f["Confidence"],
                "Issues": ", ".join(f["Issues"])
            } for f in fields
        ])
        st.dataframe(parsed_df, use_container_width=True)

    # ✏️ Right Column: Final Review
    with col2:
        st.markdown("✏️ Review and Finalize")
        for f in fields:
            label = f["Label"]
            suggestion = f.get("Suggestion")
            corrected = f["Corrected"]
            default_final = suggestion if apply_all and suggestion else corrected

            f["Final"] = st.text_input(
                f"{label} (Suggested: {suggestion or '—'})",
                value=default_final,
                key=f"final_{form_id}_{label}"
            )

            if f.get("Thumb") and f.get("Issues"):
                thumb = f["Thumb"].convert("RGB") if f["Thumb"].mode != "RGB" else f["Thumb"]
                st.image(thumb, caption=f"{label} → {', '.join(f['Issues'])}", width=220)

    flat_fields.extend([
        {
            "Form": form_id,
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

    # 🗃️ Registry Table Rows
    if form.get("TableRows"):
        st.subheader("🗃️ Registry Table Rows")
        df_table = pd.DataFrame(form["TableRows"])
        st.dataframe(df_table, use_container_width=True)

        st.download_button(
            label=f"📥 Download Table — Form {form_id}",
            data=df_table.to_csv(index=False),
            file_name=f"registry_table_form_{form_id}.csv",
            mime="text/csv"
        )

# 📤 Export Corrected Fields
st.header("📤 Export Corrected Fields")

df_export = pd.DataFrame(flat_fields)

st.download_button(
    label="📄 Download as CSV",
    data=df_export.drop(columns=["Thumb"]).to_csv(index=False),
    file_name="registry_fields.csv",
    mime="text/csv"
)

st.download_button(
    label="📄 Download as JSON",
    data=json.dumps(flat_fields, indent=2, ensure_ascii=False),
    file_name="registry_fields.json",
    mime="application/json"
)

# 📈 Confidence Summary
st.header("📈 Confidence Analysis")

if not df_export.empty:
    avg_conf = round(df_export["Confidence"].mean(), 2)
    st.markdown(f"📌 Average field confidence: **{avg_conf}%**")

    low_conf_df = df_export[df_export["Confidence"] < 50.0]
    if not low_conf_df.empty:
        st.subheader("🔍 Fields with Confidence Below 50%")
        st.dataframe(low_conf_df.drop(columns=["Thumb"]), use_container_width=True)
else:
    st.warning("⚠️ No confidence data available.")

# 🚨 Flagged Validation Issues
st.header("🚨 Validation Issues Summary")

problem_fields = [f for f in flat_fields if f["Issues"]]
if problem_fields:
    st.dataframe(pd.DataFrame(problem_fields).drop(columns=["Thumb"]), use_container_width=True)
else:
    st.markdown("✅ No flagged issues found across forms.")

# 💡 Applied Suggestions
st.header("💡 Applied Suggestions")

used_suggestions = [f for f in flat_fields if f["Suggestion"]]
if used_suggestions:
    for f in used_suggestions:
        st.markdown(f"**Form {f['Form']} — {f['Label']}**")
        st.markdown(f"🔍 Parsed: `{f['Corrected']}`")
        st.markdown(f"💡 Suggestion: `{f['Suggestion']}` → Final: `{f['Final']}`")
        st.markdown("---")
else:
    st.markdown("🟢 No suggestions applied.")

# 💾 Export Fallback Layout
st.header("📦 Fallback Box Layout Export")

st.download_button(
    label="💾 Download Box Layout as JSON",
    data=json.dumps(manual_boxes_per_form, indent=2, ensure_ascii=False),
    file_name="manual_boxes_per_form.json",
    mime="application/json"
)
