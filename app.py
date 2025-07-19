import streamlit as st
import os, json, unicodedata, re
from PIL import Image, ImageDraw
import pandas as pd
from io import BytesIO
from collections import defaultdict
from google.cloud import documentai_v1 as documentai
from pytesseract import pytesseract

# Normalize Greek text
def normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\sΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]", "", text)
    return text.upper().strip()

# Crop excess whitespace
def trim_whitespace(image, threshold=240, buffer=10):
    gray = image.convert("L")
    pixels = gray.load()
    w, h = image.size
    top = next((y for y in range(h) if any(pixels[x, y] < threshold for x in range(w))), 0)
    bottom = next((y for y in reversed(range(h)) if any(pixels[x, y] < threshold for x in range(w))), h)
    left = next((x for x in range(w) if any(pixels[x, y] < threshold for y in range(h))), 0)
    right = next((x for x in reversed(range(w)) if any(pixels[x, y] < threshold for y in range(h))), w)
    return image.crop((max(0, left - buffer), max(0, top - buffer), min(w, right + buffer), min(h, bottom + buffer)))

# Crop left side of registry
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# Split into vertical zones
def split_zones_fixed(image, overlap_px):
    w, h = image.size
    thirds = [int(h * t) for t in [0.0, 0.33, 0.66, 1.0]]
    bounds = [(thirds[0], thirds[1] + overlap_px), (thirds[1] - overlap_px, thirds[2] + overlap_px), (thirds[2] - overlap_px, thirds[3])]
    return [image.crop((0, t, w, b)) for t, b in bounds], bounds

# Parse with Document AI
def parse_docai(pil_img, project_id, processor_id, location):
    try:
        client = documentai.DocumentProcessorServiceClient(client_options={"api_endpoint": f"{location}-documentai.googleapis.com"})
        name = client.processor_path(project_id, location, processor_id)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        raw = documentai.RawDocument(content=buf.getvalue(), mime_type="image/jpeg")
        return client.process_document(request=documentai.ProcessRequest(name=name, raw_document=raw)).document
    except Exception as e:
        st.error(f"📛 Document AI Error: {e}")
        return None
# Extract fields from Document AI results
def extract_fields(doc, target_labels):
    if not doc or not doc.pages: return []
    extracted = {}
    for page in doc.pages:
        for f in page.form_fields:
            label = f.field_name.text_anchor.content or ""
            value = f.field_value.text_anchor.content or ""
            conf = round(f.field_value.confidence * 100, 2)
            label_norm = normalize(label)
            for t in target_labels:
                if label_norm == normalize(t):
                    extracted[t] = {
                        "Raw": value.strip(),
                        "Corrected": normalize(value),
                        "Confidence": conf,
                        "Schema": normalize(t)
                    }
    fields = []
    for label in target_labels:
        f = extracted.get(label, {
            "Raw": "", "Corrected": "", "Confidence": 0.0, "Schema": normalize(label)
        })
        fields.append({"Label": label, **f})
    return fields

# OCR fallback for missing fields using manual bounding boxes
def extract_field_from_box(pil_img, box):
    w, h = pil_img.size
    x, y, bw, bh = box
    x1, y1 = int(x * w), int(y * h)
    x2, y2 = int((x + bw) * w), int((y + bh) * h)
    cropped = pil_img.crop((x1, y1, x2, y2))
    raw = pytesseract.image_to_string(cropped, config="--psm 6").strip()
    return raw

# 🧭 Set up app UI
st.set_page_config(layout="wide", page_title="Greek Registry Parser")
st.title("📜 Greek Registry Parser — OCR with Manual Field Recovery")

# Sidebar Controls
overlap = st.sidebar.slider("📏 Zone Overlap", 0, 120, 60, step=10)
cred_file = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f: f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

image_file = st.file_uploader("🖼️ Upload Registry Image", type=["jpg", "jpeg", "png"])
if not image_file:
    st.info("ℹ️ Upload an image to begin.")
    st.stop()

# 🖼️ Preprocess image
image = Image.open(image_file)
image = trim_whitespace(image)
image = crop_left(image)
zones, bounds = split_zones_fixed(image, overlap)

st.image(image, caption="🖼️ Full Registry (Left Side)", use_container_width=True)

# 📋 Setup OCR Parameters
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

target_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]

# 🔧 Manual fallback boxes for unmatched fields
manual_boxes = {
    "ΟΝΟΜΑ ΠΑΤΡΟΣ": (0.05, 0.25, 0.4, 0.07),
    "ΟΝΟΜΑ ΜΗΤΡΟΣ": (0.05, 0.34, 0.4, 0.07)
}

forms_parsed = []
for idx, zone in enumerate(zones, start=1):
    st.header(f"📄 Form {idx}")
    doc = parse_docai(zone.copy(), project_id, processor_id, location)
    if not doc: continue

    fields = extract_fields(doc, target_labels)
    missing = [f["Label"] for f in fields if not f["Raw"].strip()]

    # Use OCR fallback for missing labels
    for label in missing:
        if label in manual_boxes:
            value = extract_field_from_box(zone, manual_boxes[label])
            for f in fields:
                if f["Label"] == label:
                    f["Raw"] = value
                    f["Corrected"] = normalize(value)
                    f["Confidence"] = 0.0

    st.subheader("🧾 Extracted Fields")
    st.dataframe(pd.DataFrame(fields), use_container_width=True)

    forms_parsed.append({
        "Form": idx,
        "Fields": fields,
        "Missing": [f["Label"] for f in fields if not f["Raw"].strip()]
    })
# 📦 Export Parsed Field Data
st.header("📤 Export All Parsed Forms")

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

df_fields = pd.DataFrame(flat_fields)

st.download_button("📄 Download Fields as CSV", df_fields.to_csv(index=False), "registry_fields.csv", "text/csv")
st.download_button("📄 Download Fields as JSON", json.dumps(flat_fields, indent=2, ensure_ascii=False), "registry_fields.json", "application/json")

# 📊 Summary Overview
st.header("📊 Summary Report")

valid = [f for f in forms_parsed if not f["Missing"]]
invalid = [f for f in forms_parsed if f["Missing"]]

st.markdown(f"✅ Parsed Forms with All Fields: **{len(valid)}**")
st.markdown(f"❌ Forms with Missing Fields: **{len(invalid)}**")

if invalid:
    st.subheader("❌ Missing Labels Breakdown")
    for f in invalid:
        st.markdown(f"- **Form {f['Form']}**: Missing `{', '.join(f['Missing'])}`")
