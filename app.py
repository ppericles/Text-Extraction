import streamlit as st
import os, json, unicodedata, re
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from io import BytesIO
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

# 🔠 Normalize Latin and Cyrillic to Greek
def fix_latin_greek(text):
    replacements = {
        "A": "Α", "B": "Β", "E": "Ε", "H": "Η", "K": "Κ", "M": "Μ",
        "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ", "Y": "Υ"
    }
    return "".join(replacements.get(c, c) for c in text)

def fix_cyrillic_greek(text):
    replacements = {
        "А": "Α", "В": "Β", "С": "Σ", "Е": "Ε", "Н": "Η",
        "К": "Κ", "М": "Μ", "О": "Ο", "Р": "Ρ", "Т": "Τ", "Х": "Χ"
    }
    return "".join(replacements.get(c, c) for c in text)

# 🧹 Normalize and sanitize text
def normalize(text):
    if not text:
        return ""
    text = fix_cyrillic_greek(fix_latin_greek(text))
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\sΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]", "", text)
    return text.upper().strip()

# 🧠 Flag field-level issues
def validate_registry_field(label, corrected_text, confidence):
    issues = []
    if not corrected_text:
        issues.append("Missing")

    is_numeric = label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"
    greek_chars = re.findall(r"[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ]", corrected_text or "")

    if not is_numeric:
        if len(greek_chars) < max(3, len(corrected_text) // 2):
            issues.append("Non-Greek characters")

    if len(corrected_text) < 2:
        issues.append("Too short")

    if confidence < 50.0:
        issues.append("Low confidence")

    return issues

# 💡 Suggest cleaned-up version
def suggest_fix(label, corrected_text, issues):
    if "Too short" in issues or "Non-Greek characters" in issues:
        fixed = corrected_text.title()
        if len(fixed) >= 2 and re.match(r"^[Α-ΩΆΈΉΊΌΎΏ][α-ωάέήίόύώ]{2,}", fixed):
            return fixed
    return None

# ✂️ Remove blank edges from image
def trim_whitespace(image, threshold=240, buffer=10):
    gray = image.convert("L")
    pixels = gray.load()
    w, h = image.size
    top = next((y for y in range(h) if any(pixels[x, y] < threshold for x in range(w))), 0)
    bottom = next((y for y in reversed(range(h)) if any(pixels[x, y] < threshold for x in range(w))), h)
    left = next((x for x in range(w) if any(pixels[x, y] < threshold for y in range(h))), 0)
    right = next((x for x in reversed(range(w)) if any(pixels[x, y] < threshold for y in range(h))), w)
    return image.crop((max(0, left - buffer), max(0, top - buffer), min(w, right + buffer), min(h, bottom + buffer)))

def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

def split_zones_fixed(image, overlap_px):
    w, h = image.size
    thirds = [int(h * t) for t in [0.0, 0.33, 0.66, 1.0]]
    bounds = [(thirds[0], thirds[1] + overlap_px), (thirds[1] - overlap_px, thirds[2] + overlap_px), (thirds[2] - overlap_px, thirds[3])]
    return [image.crop((0, t, w, b)) for t, b in bounds], bounds

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
        return client.process_document(request=documentai.ProcessRequest(name=name, raw_document=raw)).document
    except Exception as e:
        st.error(f"📛 Document AI error: {e}")
        return None

# 🧮 Estimate field confidence score
def estimate_confidence(label, text):
    text = text.strip()
    if not text:
        return 0.0
    if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ":
        return 90.0 if text.isdigit() else 40.0
    elif label in ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]:
        is_greekish = re.match(r"^[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ\s\-]{3,}$", text)
        return 75.0 if is_greekish else 30.0
    return 50.0

# 🩹 Vision OCR fallback with crop safety
def extract_field_from_box_with_vision(pil_img, box, label):
    try:
        x, y, bw, bh = [float(v) for v in box]
        if any(v is None for v in [x, y, bw, bh]):
            raise ValueError("Box contains None")

        w, h = pil_img.size
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + bw) * w)
        y2 = int((y + bh) * h)

        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid crop coordinates")

        cropped = pil_img.convert("RGB").crop((x1, y1, x2, y2))
        if cropped.size == (0, 0):
            raise ValueError("Empty crop region")

    except Exception as e:
        st.warning(f"⚠️ Skipping field '{label}' due to crop error: {e}")
        return "", 0.0

    try:
        buf = BytesIO()
        cropped.save(buf, format="JPEG")
        buf.seek(0)

        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=buf.read())
        image_context = {"language_hints": ["el"]}

        response = client.text_detection(image=image, image_context=image_context)
        if response.error.message:
            st.warning(f"🛑 Vision API error for '{label}': {response.error.message}")
            return "", 0.0

        desc = response.text_annotations[0].description.strip() if response.text_annotations else ""
        return desc, estimate_confidence(label, desc)

    except Exception as e:
        st.warning(f"🛑 Vision OCR failed for '{label}': {e}")
        return "", 0.0
# 🧰 Setup Streamlit page
st.set_page_config(page_title="📜 Greek Registry Parser", layout="wide")
st.title("📜 Greek Registry Parser — AI + Fallback Review")

# 🧠 Parser setup
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
target_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ",
    "ΕΠΩΝΥΜΟΝ",
    "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ",
    "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]
forms_parsed = []

# 🧩 Sidebar inputs
overlap = st.sidebar.slider("🔁 Zone Overlap", 0, 120, value=50)
normalize_input = st.sidebar.checkbox("📏 Normalize Box Inputs", value=True)

cred_file = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials Loaded")

uploaded_box_map = st.sidebar.file_uploader("📥 Import Box Map", type=["json"])
manual_boxes_per_form = {}
if uploaded_box_map:
    try:
        manual_boxes_per_form = json.load(uploaded_box_map)
        st.sidebar.success(f"✅ Loaded box map for {len(manual_boxes_per_form)} form(s)")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load box map: {e}")

# 🖼️ Image Upload
uploaded_image = st.file_uploader("🖼️ Upload Registry Image", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("ℹ️ Please upload a registry image to continue.")
    st.stop()

# ✂️ Trim and crop
try:
    original = Image.open(uploaded_image)
    cropped = crop_left(trim_whitespace(original))
    if cropped.size == (0, 0):
        st.error("❌ Cropped image is empty.")
        st.stop()
    zones, bounds = split_zones_fixed(cropped, overlap)
    if not zones:
        st.error("❌ Failed to split image into zones.")
        st.stop()
except Exception as e:
    st.error(f"❌ Image processing error: {e}")
    st.stop()

# 🖼️ Preview cropped image
st.image(cropped, caption="🖼️ Cropped Registry (Left Half)", use_container_width=True)
st.header("🖼️ Zone Previews")
for i, zone in enumerate(zones, start=1):
    st.image(zone, caption=f"Zone {i}", width=300)

# 🧾 Parse each form zone
for idx, zone in enumerate(zones, start=1):
    st.header(f"📄 Form {idx}")
    zone_w, zone_h = zone.size

    # 📝 Box editor
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

    # ↔️ Propagate Form 1 layout
    if idx == 1:
        layout_1 = manual_boxes_per_form["1"]
        for z in range(2, len(zones) + 1):
            fid = str(z)
            if fid not in manual_boxes_per_form:
                manual_boxes_per_form[fid] = layout_1
        st.info("📐 Applied Form 1 layout to remaining zones")

    # 🎨 Draw overlays
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
            st.warning(f"⚠️ Overlay issue for '{label}': {e}")
    st.image(overlay, caption=f"🟣 Fallback Boxes — Form {idx}", use_container_width=True)

    # 🔍 Document AI parse
    doc = parse_docai(zone.copy(), project_id, processor_id, location)
    extracted = {}
    if doc:
        for page in doc.pages:
            for f in page.form_fields:
                raw_label = f.field_name.text_anchor.content or ""
                raw_value = f.field_value.text_anchor.content or ""
                conf = round(f.field_value.confidence * 100, 2)
                for target in target_labels:
                    if normalize(raw_label) == normalize(target):
                        corrected = normalize(fix_cyrillic_greek(fix_latin_greek(raw_value)))
                        issues = validate_registry_field(target, corrected, conf)
                        suggestion = suggest_fix(target, corrected, issues)
                        extracted[target] = {
                            "Label": target,
                            "Raw": raw_value.strip(),
                            "Corrected": corrected,
                            "Confidence": conf,
                            "Issues": issues,
                            "Suggestion": suggestion,
                            "Thumb": None
                        }

    # 🩹 Vision fallback and thumbnails
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

    forms_parsed.append({
        "Form": idx,
        "Fields": fields,
        "Missing": [f["Label"] for f in fields if not f["Raw"].strip()]
    })

# ✅ Done parsing
st.success("✅ All registry zones parsed successfully")
# 🧠 Reviewer Tools
st.header("📊 Registry Review and Export Dashboard")

apply_all = st.checkbox("🧠 Apply All Suggested Corrections Automatically", value=False)
optional_labels = ["ΟΝΟΜΑ ΜΗΤΡΟΣ"]  # Modify this list if needed
flat_fields = []

for form in forms_parsed:
    form_id = form["Form"]
    fields = form["Fields"]

    st.subheader(f"📄 Form {form_id}")
    col1, col2 = st.columns(2)

    # 📝 Parsed values and confidence
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

    # ✏️ Reviewer edits
    with col2:
        st.markdown("✏️ Review & Finalize")
        for f in fields:
            label = f["Label"]
            suggested = f.get("Suggestion")
            corrected = f["Corrected"]

            default_final = (
                suggested if apply_all and suggested else corrected
            )
            f["Final"] = st.text_input(
                f"{label} (Suggestion: {suggested or '—'})",
                value=default_final,
                key=f"final_{form_id}_{label}"
            )

            # 🖼️ Thumbnail for flagged fields
            thumb = f.get("Thumb")
            if thumb and f.get("Issues"):
                if thumb.mode != "RGB":
                    thumb = thumb.convert("RGB")
                st.image(thumb, caption=f"{label} → {', '.join(f['Issues'])}", width=200)

    # 📦 Aggregate for export
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

# 📤 Export Results
st.header("📤 Export Corrected Data")

df_export = pd.DataFrame(flat_fields)

st.download_button(
    label="📄 Download as CSV",
    data=df_export.drop(columns=["Thumb"]).to_csv(index=False),
    file_name="registry_corrected.csv",
    mime="text/csv"
)

st.download_button(
    label="📄 Download as JSON",
    data=json.dumps(flat_fields, indent=2, ensure_ascii=False),
    file_name="registry_corrected.json",
    mime="application/json"
)

# 📈 Confidence Analytics
st.header("📈 Confidence Overview")

if not df_export.empty:
    avg_conf = round(df_export["Confidence"].mean(), 2)
    st.markdown(f"📌 Average confidence across fields: **{avg_conf}%**")

    low_conf = df_export[df_export["Confidence"] < 50.0]
    if not low_conf.empty:
        st.subheader("🔍 Low-confidence Fields (< 50%)")
        st.dataframe(low_conf.drop(columns=["Thumb"]), use_container_width=True)
else:
    st.warning("⚠️ No field data available.")

# 🚨 Validation Summary
st.header("🚨 Flagged Issues Summary")

problem_fields = [f for f in flat_fields if f["Issues"]]
if problem_fields:
    st.dataframe(pd.DataFrame(problem_fields).drop(columns=["Thumb"]), use_container_width=True)
else:
    st.markdown("✅ No validation issues detected.")

# 💡 Applied Suggestions
st.header("💡 Suggestions Applied")

used_suggestions = [f for f in flat_fields if f["Suggestion"]]
if used_suggestions:
    for f in used_suggestions:
        st.markdown(f"**Form {f['Form']} — {f['Label']}**")
        st.markdown(f"🔍 Corrected: `{f['Corrected']}`")
        st.markdown(f"💡 Suggestion: `{f['Suggestion']}` → Final: `{f['Final']}`")
        st.markdown("---")
else:
    st.markdown("🟢 No suggestions were used.")

# 💾 Fallback Layout Export
st.header("📦 Fallback Box Layout")

st.download_button(
    label="💾 Download Box Layout (JSON)",
    data=json.dumps(manual_boxes_per_form, indent=2, ensure_ascii=False),
    file_name="manual_boxes_per_form.json",
    mime="application/json"
)
