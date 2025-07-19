import streamlit as st
import os, json, unicodedata, re
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from io import BytesIO
import numpy as np
from google.cloud import documentai_v1 as documentai
from google.cloud import vision
from streamlit_drawable_canvas import st_canvas

# Normalize Greek text
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

def extract_field_from_box_with_vision(pil_img, box):
    w, h = pil_img.size
    x, y, bw, bh = box
    x1, y1 = int(x * w), int(y * h)
    x2, y2 = int((x + bw) * w), int((y + bh) * h)
    cropped = pil_img.crop((x1, y1, x2, y2))
    buf = BytesIO()
    cropped.save(buf, format="JPEG")
    buf.seek(0)

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=buf.read())
    response = client.text_detection(image=image)

    if response.error.message:
        st.warning(f"🛑 Vision API Error: {response.error.message}")
        return ""
    return response.text_annotations[0].description.strip() if response.text_annotations else ""

def draw_fallback_boxes(img, labels, box_map):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except:
        font = None
    for label in labels:
        if label in box_map:
            x, y, bw, bh = box_map[label]
            x1, y1 = int(x * w), int(y * h)
            x2, y2 = int((x + bw) * w), int((y + bh) * h)
            draw.rectangle([(x1, y1), (x2, y2)], outline="purple", width=2)
            draw.text((x1, y1 - 16), label, fill="purple", font=font or None)
    return img
# Title & layout
st.set_page_config(layout="wide", page_title="Registry Parser")
st.title("📜 Greek Registry Parser — AI + Manual Fallback")

# Sidebar controls
overlap = st.sidebar.slider("🔁 Form Zone Overlap", 0, 120, 60)
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

show_fallback_boxes = st.sidebar.checkbox("🟣 Show fallback overlays", value=True)

uploaded_image = st.file_uploader("🖼️ Upload Registry Image", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.info("Please upload a registry image to begin.")
    st.stop()

# Image preprocessing
original = Image.open(uploaded_image)
cropped = crop_left(trim_whitespace(original))
zones, bounds = split_zones_fixed(cropped, overlap)
st.image(cropped, caption="📘 Trimmed Registry (Left Side)", use_container_width=True)

# OCR config
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

target_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]

forms_parsed = []

# Parse each form zone
for idx, zone in enumerate(zones, start=1):
    st.header(f"📄 Form {idx}")
    doc = parse_docai(zone.copy(), project_id, processor_id, location)
    if not doc: continue

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

    # Vision fallback
    form_boxes = manual_boxes_per_form.get(str(idx), {})
    fields = []
    for label in target_labels:
        if label in extracted and extracted[label]["Raw"].strip():
            fields.append({
                "Label": label,
                "Raw": extracted[label]["Raw"],
                "Corrected": extracted[label]["Corrected"],
                "Confidence": extracted[label]["Confidence"]
            })
        else:
            fallback_text = ""
            if label in form_boxes:
                fallback_text = extract_field_from_box_with_vision(zone, form_boxes[label])
            fields.append({
                "Label": label,
                "Raw": fallback_text,
                "Corrected": normalize(fallback_text),
                "Confidence": 0.0
            })

    st.subheader("🧾 Extracted Fields")
    st.dataframe(pd.DataFrame(fields), use_container_width=True)

    # Show overlay boxes
    if show_fallback_boxes:
        fallback_labels = [f["Label"] for f in fields if f["Confidence"] == 0.0 and f["Raw"]]
        overlay = draw_fallback_boxes(zone.copy(), fallback_labels, form_boxes)
        st.image(overlay, caption="🖼️ Vision AI Fallback Boxes", use_container_width=True)

    # Canvas drawing (safe array conversion)
    st.subheader("✍️ Manual Fallback Box Drawing")
    try:
        zone_array = np.array(zone.convert("RGB"))
        assert zone_array.ndim == 3 and zone_array.shape[2] == 3
        assert zone_array.size > 0
    except Exception as e:
        st.error(f"🛑 Canvas error for Form {idx}: {e}")
        continue

    canvas_result = st_canvas(
        background_image=zone_array,
        update_streamlit=True,
        height=zone_array.shape[0],
        width=zone_array.shape[1],
        drawing_mode="rect",
        stroke_color="purple",
        key=f"canvas_{idx}",
        display_toolbar=True
    )

    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        manual_boxes_per_form[str(idx)] = {}
        for obj in canvas_result.json_data["objects"]:
            label = obj.get("label") or f"Box_{idx}"
            x = obj["left"] / zone_array.shape[1]
            y = obj["top"] / zone_array.shape[0]
            w = obj["width"] / zone_array.shape[1]
            h = obj["height"] / zone_array.shape[0]
            manual_boxes_per_form[str(idx)][label] = (x, y, w, h)
        st.success(f"✅ Saved {len(manual_boxes_per_form[str(idx)])} box(es) for Form {idx}")

    forms_parsed.append({
        "Form": idx,
        "Fields": fields,
        "Missing": [f["Label"] for f in fields if not f["Raw"].strip()]
    })
# 📤 Export Parsed Fields
st.header("📤 Export Parsed Fields")

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
    label="📄 Download as CSV",
    data=df.to_csv(index=False),
    file_name="registry_fields.csv",
    mime="text/csv"
)

st.download_button(
    label="📄 Download as JSON",
    data=json.dumps(flat_fields, indent=2, ensure_ascii=False),
    file_name="registry_fields.json",
    mime="application/json"
)

# 📊 Summary Report
st.header("📊 Form Parsing Summary")

valid_forms = [f for f in forms_parsed if not f["Missing"]]
invalid_forms = [f for f in forms_parsed if f["Missing"]]

st.markdown(f"✅ Fully parsed forms: **{len(valid_forms)}**")
st.markdown(f"❌ Forms with missing fields: **{len(invalid_forms)}**")

if invalid_forms:
    st.subheader("🚨 Missing Fields Breakdown")
    for f in invalid_forms:
        st.markdown(f"- **Form {f['Form']}** → Missing: `{', '.join(f['Missing'])}`")

# 💾 Export Fallback Box Layouts
if manual_boxes_per_form:
    st.header("🧠 Export Fallback Box Maps")
    st.json(manual_boxes_per_form)

    st.download_button(
        label="💾 Download Box Layouts as JSON",
        data=json.dumps(manual_boxes_per_form, indent=2, ensure_ascii=False),
        file_name="manual_boxes_per_form.json",
        mime="application/json"
    )
