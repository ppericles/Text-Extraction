import streamlit as st
import os, json, unicodedata, re
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from io import BytesIO
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

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

def normalize(text):
    if not text: return ""
    text = fix_cyrillic_greek(fix_latin_greek(text))
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\sΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]", "", text)
    return text.upper().strip()

def validate_registry_field(label, corrected_text, confidence):
    issues = []
    if not corrected_text:
        issues.append("Missing")
    greek_chars = re.findall(r"[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ]", corrected_text)
    if len(greek_chars) < max(3, len(corrected_text) // 2):
        issues.append("Non-Greek characters")
    if len(corrected_text) < 2:
        issues.append("Too short")
    if confidence < 50.0:
        issues.append("Low confidence")
    return issues

def suggest_fix(label, corrected_text, issues):
    if "Too short" in issues or "Non-Greek characters" in issues:
        fixed = corrected_text.title()
        if len(fixed) >= 2 and re.match(r"^[Α-ΩΆΈΉΊΌΎΏ][α-ωάέήίόύώ]{2,}", fixed):
            return fixed
    return None

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
        st.error(f"📛 Document AI Error: {e}")
        return None

def estimate_confidence(label, text):
    text = text.strip()
    if not text: return 0.0
    if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ":
        return 90.0 if text.isdigit() else 40.0
    elif label in ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]:
        is_greekish = re.match(r"^[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ\s\-]{3,}$", text)
        return 75.0 if is_greekish else 30.0
    return 50.0

def extract_field_from_box_with_vision(pil_img, box, label):
    try:
        x, y, bw, bh = [float(v) for v in box]
        if any(val is None for val in (x, y, bw, bh)):
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

    buf = BytesIO()
    try:
        cropped.convert("RGB").save(buf, format="JPEG")
    except Exception as e:
        st.warning(f"🛑 Could not save crop for '{label}': {e}")
        return "", 0.0

    buf.seek(0)
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=buf.read())
    image_context = {"language_hints": ["el"]}

    try:
        response = client.text_detection(image=image, image_context=image_context)
        if response.error.message:
            st.warning(f"🛑 Vision API Error for '{label}': {response.error.message}")
            return "", 0.0
        desc = response.text_annotations[0].description.strip() if response.text_annotations else ""
    except Exception as e:
        st.warning(f"🛑 Vision API request failed for '{label}': {e}")
        return "", 0.0

    return desc, estimate_confidence(label, desc)
# 📦 App config and layout
st.set_page_config(layout="wide", page_title="📜 Greek Registry Parser")
st.title("📜 Greek Registry Parser — AI + OCR Fallbacks")

# 🔧 Sidebar setup
overlap = st.sidebar.slider("🔁 Zone Overlap", 0, 120, 50)
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
        st.sidebar.success(f"✅ Loaded fallback layout for {len(manual_boxes_per_form)} form(s)")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load box map: {e}")

normalize_input = st.sidebar.checkbox("📏 Normalize box dimensions", value=True)

# 🖼️ Upload registry image
uploaded_image = st.file_uploader("🖼️ Upload Registry Image", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("ℹ️ Please upload a registry image to begin.")
    st.stop()

try:
    original = Image.open(uploaded_image)
    cropped = crop_left(trim_whitespace(original))
    if cropped.size == (0, 0):
        st.error("❌ Cropped image is empty. Check image quality.")
        st.stop()
    zones, bounds = split_zones_fixed(cropped, overlap)
    if not zones:
        st.error("❌ Failed to split registry into zones.")
        st.stop()
except Exception as e:
    st.error(f"❌ Image processing error: {e}")
    st.stop()

# 🖼️ Preview cropped image and zones
st.image(cropped, caption="🖼️ Cropped Registry (Left Side)", use_container_width=True)
st.header("🖼️ Registry Zones Preview")
for idx, zone in enumerate(zones, start=1):
    st.image(zone, caption=f"Zone {idx}", width=300)
# 🧠 Parser setup
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
target_labels = ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]
forms_parsed = []

for idx, zone in enumerate(zones, start=1):
    st.header(f"📄 Form {idx}")
    zone_width, zone_height = zone.size

    # 📝 Fallback box editor
    existing = manual_boxes_per_form.get(str(idx), {})
    rows = []
    for label in target_labels:
        box = existing.get(label, (None, None, None, None))
        try:
            x_raw, y_raw, w_raw, h_raw = [float(v) for v in box]
            x, y, w, h = (x_raw / zone_width, y_raw / zone_height, w_raw / zone_width, h_raw / zone_height) if normalize_input else (x_raw, y_raw, w_raw, h_raw)
        except:
            x, y, w, h = None, None, None, None
        rows.append({"Label": label, "X": x, "Y": y, "Width": w, "Height": h})

    box_editor = st.data_editor(
        pd.DataFrame(rows),
        use_container_width=True,
        num_rows="dynamic",
        key=f"editor_{idx}"
    )

    manual_boxes_per_form[str(idx)] = {
        row["Label"]: (row["X"], row["Y"], row["Width"], row["Height"])
        for _, row in box_editor.iterrows()
        if all(val is not None for val in (row["X"], row["Y"], row["Width"], row["Height"]))
    }

    # ↔️ Propagate Form 1 layout
    if idx == 1:
        layout_1 = manual_boxes_per_form["1"]
        for z in range(2, len(zones) + 1):
            if not manual_boxes_per_form.get(str(z)):
                manual_boxes_per_form[str(z)] = layout_1
        st.info("📐 Applied Form 1 layout to other forms")

    # 🎨 Overlay fallback boxes
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
            draw.text((x1, y1 - 16), label, fill="purple", font=font or None)
        except:
            continue
    st.image(overlay, caption=f"🟣 Annotated Boxes — Form {idx}", use_container_width=True)

    # 📄 Parse with Document AI
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

    # 🩹 Fallback parsing with Vision + thumbnail
    fields = []
    for label in target_labels:
        f = extracted.get(label)
        if f and f["Raw"]:
            fields.append(f)
            continue

        box = manual_boxes_per_form[str(idx)].get(label)
        fallback_text, confidence = extract_field_from_box_with_vision(zone, box, label) if box else ("", 0.0)
        corrected = normalize(fix_cyrillic_greek(fix_latin_greek(fallback_text)))
        issues = validate_registry_field(label, corrected, confidence)
        suggestion = suggest_fix(label, corrected, issues)

        # ✂️ Extract thumbnail
        thumb = None
        try:
            x, y, bw, bh = [float(v) for v in box]
            w, h = zone.size
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + bw) * w)
            y2 = int((y + bh) * h)
            thumb = zone.crop((x1, y1, x2, y2))
        except:
            thumb = None

        fields.append({
            "Label": label,
            "Raw": fallback_text,
            "Corrected": corrected,
            "Confidence": confidence,
            "Issues": issues,
            "Suggestion": suggestion,
            "Thumb": thumb
        })

    forms_parsed.append({
        "Form": idx,
        "Fields": fields,
        "Missing": [f["Label"] for f in fields if not f["Raw"].strip()]
    })

# 🟢 Proceed to review/export dashboard
st.success("✅ All registry zones parsed successfully")
# 📊 Registry Review & Export Dashboard
st.header("📊 Registry Review and Export Dashboard")

flat_fields = []

for form in forms_parsed:
    form_id = form["Form"]
    fields = form["Fields"]

    st.subheader(f"📄 Form {form_id}")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("📝 Parsed Field Details")
        df_parsed = pd.DataFrame([
            {
                "Label": f["Label"],
                "Raw": f["Raw"],
                "Corrected": f["Corrected"],
                "Confidence": f["Confidence"],
                "Issues": ", ".join(f.get("Issues", []))
            } for f in fields
        ])
        st.dataframe(df_parsed, use_container_width=True)

    with col2:
        st.markdown("💡 Finalize & Apply Corrections")
        for f in fields:
            label = f["Label"]
            suggestion = f.get("Suggestion")
            corrected = f["Corrected"]
            confidence = f["Confidence"]
            default_final = suggestion if suggestion else corrected

            f["Final"] = st.text_input(
                f"✏️ {label} (Suggested: {suggestion or '—'})",
                value=default_final,
                key=f"final_{form_id}_{label}"
            )

            thumb = f.get("Thumb")
            if thumb and thumb.size != (0, 0):
                if thumb.mode != "RGB":
                    thumb = thumb.convert("RGB")
                st.image(thumb, caption=f"{label} → {', '.join(f.get('Issues', []))}", width=200)
            elif f.get("Issues"):
                st.warning(f"⚠️ Thumbnail unavailable for {label}")

    flat_fields.extend([
        {
            "Form": form_id,
            "Label": f["Label"],
            "Raw": f["Raw"],
            "Corrected": f["Corrected"],
            "Final": f.get("Final", f["Corrected"]),
            "Confidence": f["Confidence"],
            "Issues": f.get("Issues", []),
            "Suggestion": f.get("Suggestion"),
            "Thumb": "Included" if f.get("Thumb") else "None"
        }
        for f in fields
    ])

# 📤 Export Section
st.header("📤 Export Reviewed Data")

df_export = pd.DataFrame(flat_fields)

st.download_button(
    label="📄 Download as CSV",
    data=df_export.drop(columns=["Thumb"]).to_csv(index=False),
    file_name="registry_reviewed.csv",
    mime="text/csv"
)

st.download_button(
    label="📄 Download as JSON",
    data=json.dumps(flat_fields, indent=2, ensure_ascii=False),
    file_name="registry_reviewed.json",
    mime="application/json"
)

# 📈 Confidence Summary
st.header("📈 Confidence Overview")

if not df_export.empty:
    avg_conf = round(df_export["Confidence"].mean(), 2)
    st.markdown(f"📌 Average field confidence: **{avg_conf}%**")

    low_conf_df = df_export[df_export["Confidence"] < 50.0]
    if not low_conf_df.empty:
        st.subheader("🔍 Fields with Low Confidence (< 50%)")
        st.dataframe(low_conf_df.drop(columns=["Thumb"]), use_container_width=True)
else:
    st.warning("⚠️ No field data available.")

# 🚨 Validation Issues Summary
st.header("🚨 Validation Issues")

problematic = [f for f in flat_fields if f.get("Issues")]
if problematic:
    st.dataframe(pd.DataFrame(problematic).drop(columns=["Thumb"]), use_container_width=True)
else:
    st.markdown("✅ No issues flagged.")

# 💡 Suggestions Review
st.header("💡 Suggested Corrections Applied")

suggested = [f for f in flat_fields if f.get("Suggestion")]
if suggested:
    for f in suggested:
        st.markdown(f"**Form {f['Form']} — {f['Label']}**")
        st.markdown(f"🔍 Corrected: `{f['Corrected']}`")
        st.markdown(f"💡 Suggestion Applied: `{f['Final']}`")
        st.markdown("---")
else:
    st.markdown("🟢 No suggestions applied.")

# 💾 Fallback Box Layout Export
if manual_boxes_per_form:
    st.header("📦 Fallback Box Layout Map")
    st.json(manual_boxes_per_form)
    st.download_button(
        label="💾 Download Box Map as JSON",
        data=json.dumps(manual_boxes_per_form, indent=2, ensure_ascii=False),
        file_name="manual_boxes_per_form.json",
        mime="application/json"
    )
