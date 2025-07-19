import streamlit as st
import os, json, unicodedata, re
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from io import BytesIO
from google.cloud import documentai_v1 as documentai
from google.cloud import vision

def normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\sΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]", "", text)
    return text.upper().strip()

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
    except Exception as e:
        st.warning(f"⚠️ Invalid box values for '{label}': {box} — {e}")
        return "", 0.0

    w, h = pil_img.size
    try:
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + bw) * w)
        y2 = int((y + bh) * h)
    except Exception as e:
        st.warning(f"⚠️ Could not compute crop for '{label}': {box} — {e}")
        return "", 0.0

    zone_rgb = pil_img.convert("RGB")
    cropped = zone_rgb.crop((x1, y1, x2, y2)).copy()

    buf = BytesIO()
    try:
        cropped.save(buf, format="JPEG")
    except Exception as e:
        st.warning(f"🛑 Could not save crop for '{label}': {e}")
        return "", 0.0

    buf.seek(0)
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=buf.read())
    response = client.text_detection(image=image)

    if response.error.message:
        st.warning(f"🛑 Vision API Error for '{label}': {response.error.message}")
        return "", 0.0

    desc = response.text_annotations[0].description.strip() if response.text_annotations else ""
    return desc, estimate_confidence(label, desc)
# App layout & sidebar controls
st.set_page_config(layout="wide", page_title="Registry Parser")
st.title("📜 Greek Registry Parser — AI + Manual Fallbacks")

overlap = st.sidebar.slider("🔁 Zone Overlap", 0, 120, 50)
cred_file = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f: f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

uploaded_box_map = st.sidebar.file_uploader("📥 Import Fallback Box Map", type=["json"])
manual_boxes_per_form = {}
if uploaded_box_map:
    try:
        manual_boxes_per_form = json.load(uploaded_box_map)
        st.sidebar.success(f"✅ Loaded box map for {len(manual_boxes_per_form)} form(s)")
    except Exception as e:
        st.sidebar.error(f"📛 Failed to load box map: {e}")

normalize_input = st.checkbox("📏 Normalize pixel box map automatically", value=True)

uploaded_image = st.file_uploader("🖼️ Upload Registry Image", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("ℹ️ Please upload a registry image to continue.")
    st.stop()

# Preprocessing
original = Image.open(uploaded_image)
cropped = crop_left(trim_whitespace(original))
zones, bounds = split_zones_fixed(cropped, overlap)
st.image(cropped, caption="🖼️ Trimmed Registry (Left Side)", use_container_width=True)

project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
target_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]

forms_parsed = []

# Loop through zones
for idx, zone in enumerate(zones, start=1):
    st.header(f"📄 Form {idx}")
    zone_width, zone_height = zone.size

    # Prefill fallback box editor
    existing = manual_boxes_per_form.get(str(idx), {})
    prefill_rows = []
    for label in target_labels:
        box = existing.get(label, (None, None, None, None))
        try:
            x_raw, y_raw, w_raw, h_raw = [float(v) for v in box]
            normalized = all(0.0 <= val <= 1.0 for val in (x_raw, y_raw, w_raw, h_raw))
            if normalize_input and not normalized:
                x = x_raw / zone_width
                y = y_raw / zone_height
                w = w_raw / zone_width
                h = h_raw / zone_height
            else:
                x, y, w, h = x_raw, y_raw, w_raw, h_raw
            if any(val is None or val <= 0 or val > 1 for val in (x, y, w, h)):
                raise ValueError("Invalid box dimensions")
        except Exception:
            x, y, w, h = None, None, None, None
        prefill_rows.append({"Label": label, "X": x, "Y": y, "Width": w, "Height": h})

    box_editor = st.data_editor(
        pd.DataFrame(prefill_rows),
        use_container_width=True,
        num_rows="dynamic",
        key=f"editor_{idx}"
    )

    # Store updated boxes
    manual_boxes_per_form[str(idx)] = {}
    for _, row in box_editor.iterrows():
        label = row["Label"]
        x, y, w, h = row["X"], row["Y"], row["Width"], row["Height"]
        if all(val is not None for val in (x, y, w, h)):
            manual_boxes_per_form[str(idx)][label] = (x, y, w, h)

    # Overlay fallback boxes
    overlay = zone.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except:
        font = None

    for label, box in manual_boxes_per_form.get(str(idx), {}).items():
        try:
            x, y, bw, bh = [float(v) for v in box]
            if any(val is None for val in (x, y, bw, bh)):
                raise ValueError("Box contains None")
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + bw) * w)
            y2 = int((y + bh) * h)
            draw.rectangle([(x1, y1), (x2, y2)], outline="purple", width=2)
            draw.text((x1, y1 - 16), label, fill="purple", font=font or None)
        except Exception as e:
            st.warning(f"⚠️ Skipping '{label}' — {e}")
    st.image(overlay, caption=f"🟣 Fallback Boxes for Form {idx}", use_container_width=True)

    # Document AI parse
    doc = parse_docai(zone.copy(), project_id, processor_id, location)
    if not doc:
        st.error(f"❌ No Document AI result for Form {idx}")
        continue

    extracted = {}
    for page in doc.pages:
        for f in page.form_fields:
            label = f.field_name.text_anchor.content or ""
            value = f.field_value.text_anchor.content or ""
            conf = round(f.field_value.confidence * 100, 2)
            for t in target_labels:
                if normalize(label) == normalize(t):
                    extracted[t] = {
                        "Raw": value.strip(),
                        "Corrected": normalize(value),
                        "Confidence": conf
                    }

    # Merge AI & fallback
    fields = []
    for label in target_labels:
        if label in extracted and extracted[label]["Raw"]:
            fields.append({
                "Label": label,
                "Raw": extracted[label]["Raw"],
                "Corrected": extracted[label]["Corrected"],
                "Confidence": extracted[label]["Confidence"]
            })
        else:
            box = manual_boxes_per_form[str(idx)].get(label)
            fallback_text, confidence = extract_field_from_box_with_vision(zone, box, label) if box else ("", 0.0)
            fields.append({
                "Label": label,
                "Raw": fallback_text,
                "Corrected": normalize(fallback_text),
                "Confidence": confidence
            })

    st.subheader("🧾 Parsed Fields")
    st.dataframe(pd.DataFrame(fields), use_container_width=True)

    forms_parsed.append({
        "Form": idx,
        "Fields": fields,
        "Missing": [f["Label"] for f in fields if not f["Raw"].strip()]
    })
# 📤 Export Parsed Field Data
st.header("📤 Export Parsed Field Data")

flat_fields = []
for form in forms_parsed:
    for field in form["Fields"]:
        flat_fields.append({
            "Form": form["Form"],
            "Label": field["Label"],
            "Raw": field["Raw"],
            "Corrected": field["Corrected"],
            "Confidence": field["Confidence"]
        })

df = pd.DataFrame(flat_fields)

st.download_button(
    label="📄 Download Fields as CSV",
    data=df.to_csv(index=False),
    file_name="registry_fields.csv",
    mime="text/csv"
)

st.download_button(
    label="📄 Download Fields as JSON",
    data=json.dumps(flat_fields, indent=2, ensure_ascii=False),
    file_name="registry_fields.json",
    mime="application/json"
)

# 📊 Parsing Summary
st.header("📊 Parsing Summary")

valid_forms = [f for f in forms_parsed if not f["Missing"]]
invalid_forms = [f for f in forms_parsed if f["Missing"]]

st.markdown(f"✅ Fully parsed forms: **{len(valid_forms)}**")
st.markdown(f"❌ Forms with missing fields: **{len(invalid_forms)}**")

if invalid_forms:
    st.subheader("🚨 Missing Fields Breakdown")
    for f in invalid_forms:
        missing = ", ".join(f["Missing"])
        st.markdown(f"- **Form {f['Form']}** → Missing: `{missing}`")

# 📈 Confidence Overview
st.header("📈 Confidence Overview")

if not df.empty:
    avg_conf = round(df["Confidence"].mean(), 2)
    st.markdown(f"📌 Average confidence across all fields: **{avg_conf}%**")

    low_conf_fields = df[df["Confidence"] < 50.0]
    if not low_conf_fields.empty:
        st.subheader("🔍 Fields with Low Confidence (< 50%)")
        st.dataframe(low_conf_fields, use_container_width=True)
else:
    st.markdown("⚠️ No field data available to summarize.")

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
