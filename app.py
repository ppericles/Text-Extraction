import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os, json, unicodedata
import pandas as pd
import re
from io import BytesIO
from collections import defaultdict
from google.cloud import documentai_v1 as documentai
from streamlit_drawable_canvas import st_canvas

def normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFD", text)
    return ''.join(c for c in text if unicodedata.category(c) != "Mn").upper().strip()

def trim_whitespace(image, threshold=240, buffer=10):
    gray = image.convert("L")
    w, h = gray.size
    pixels = gray.load()
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
    bounds = [
        (thirds[0], thirds[1] + overlap_px),
        (thirds[1] - overlap_px, thirds[2] + overlap_px),
        (thirds[2] - overlap_px, thirds[3])
    ]
    return [image.crop((0, t, w, b)) for t, b in bounds], bounds

def show_zone_overlay(image, bounds):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for top, bottom in bounds:
        draw.rectangle([(0, top), (image.width, bottom)], outline="red", width=3)
    return preview

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

def extract_fields(doc, labels):
    extracted = {}
    items = [
        {
            "Label": f.field_name.text_anchor.content or "",
            "Value": f.field_value.text_anchor.content or "",
            "Confidence": round(f.field_value.confidence * 100, 2)
        }
        for page in doc.pages for f in page.form_fields
    ]
    for item in items:
        label = item["Label"]
        if normalize(label) in [normalize(l) for l in labels]:
            extracted[label] = {
                "Raw": item["Value"].strip(),
                "Corrected": normalize(item["Value"]),
                "Confidence": item["Confidence"],
                "Schema": normalize(label)
            }
    return [
        {"Label": label, **extracted.get(label, {
            "Raw": "", "Corrected": "", "Confidence": 0.0, "Schema": normalize(label)
        })}
        for label in labels
    ]

def learn_field_positions(doc, labels):
    positions = defaultdict(list)
    for page in doc.pages:
        for f in page.form_fields:
            label = f.field_name.text_anchor.content or ""
            if normalize(label) not in [normalize(t) for t in labels]: continue
            layout = getattr(f.field_value, "layout", None)
            if layout and getattr(layout, "bounding_poly", None) and layout.bounding_poly.normalized_vertices:
                verts = layout.bounding_poly.normalized_vertices
                xs = [v.x for v in verts if v]
                ys = [v.y for v in verts if v]
                if xs and ys:
                    x, y = min(xs), min(ys)
                    w, h = max(xs) - x, max(ys) - y
                    positions[label].append((x, y, w, h))
    return positions

def average_positions(position_dict):
    return {
        label: (
            sum(x)/len(x),
            sum(y)/len(y),
            sum(w)/len(w),
            sum(h)/len(h)
        )
        for label, boxes in position_dict.items() if boxes
    }

def overlay_missing_labels(img, missing_labels, learned, fallback, custom):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except:
        font = None
    for label in missing_labels:
        box = custom.get(label) or learned.get(label) or fallback.get(label)
        if not box: continue
        x, y, bw, bh = box
        draw.rectangle([(int(x*w), int(y*h)), (int((x+bw)*w), int((y+bh)*h))], outline="orange", width=2)
        draw.text((int(x*w), int(y*h) - 16), label, fill="orange", font=font)
    return img
# UI Setup
st.set_page_config(layout="wide", page_title="Greek Registry Parser")
st.title("🏛️ Registry OCR — Manual Calibration & Adaptive Overlay")

# Sidebar Controls
overlap = st.sidebar.slider("🧩 Overlap between zones", 0, 120, 60, 10)
show_overlay = st.sidebar.checkbox("🟧 Show missing field overlays", value=True)
calibrate_mode = st.sidebar.checkbox("🎯 Calibrate label positions manually")

cred_file = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

uploaded_calibration = st.sidebar.file_uploader("📥 Load saved field boxes", type=["json"])
custom_positions = {}

if uploaded_calibration:
    try:
        custom_positions = json.loads(uploaded_calibration.read().decode("utf-8"))
        st.sidebar.success(f"✅ Loaded {len(custom_positions)} custom field boxes")
    except Exception as e:
        st.sidebar.error(f"📛 Failed to load calibration file: {e}")

file = st.file_uploader("📎 Upload Registry Image", type=["jpg", "jpeg", "png"])
if not file:
    st.info("ℹ️ Upload an image to begin")
    st.stop()

image = Image.open(file)
trimmed = trim_whitespace(image)
img_left = crop_left(trimmed)
zones, bounds = split_zones_fixed(img_left, overlap_px=overlap)

preview = show_zone_overlay(img_left, bounds)
st.image(preview, caption="📐 Zone Preview", use_container_width=True)

# OCR Setup
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
target_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]

fallback_regions = {
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": (0.05, 0.06, 0.4, 0.08),
    "ΕΠΩΝΥΜΟΝ":         (0.05, 0.16, 0.4, 0.07),
    "ΟΝΟΜΑ ΠΑΤΡΟΣ":     (0.05, 0.25, 0.4, 0.07),
    "ΟΝΟΜΑ ΜΗΤΡΟΣ":     (0.05, 0.34, 0.4, 0.07),
    "ΚΥΡΙΟΝ ΟΝΟΜΑ":     (0.05, 0.43, 0.4, 0.07)
}

parsed_forms = []
learned_field_positions = defaultdict(list)

# 🧠 Form Loop
for i, zone_img in enumerate(zones, start=1):
    st.header(f"📄 Form {i}")
    doc = parse_docai(zone_img.copy(), project_id, processor_id, location)
    if not doc: continue

    fields = extract_fields(doc, target_labels)
    new_positions = learn_field_positions(doc, target_labels)
    for label, boxes in new_positions.items():
        learned_field_positions[label].extend(boxes)

    found_labels = [f["Label"] for f in fields if f["Raw"].strip()]
    missing_labels = [label for label in target_labels if label not in found_labels]

    st.subheader("🕵️ Field Label Report")
    st.markdown(f"✅ Found: `{', '.join(found_labels)}`")
    st.markdown(f"❌ Missing: `{', '.join(missing_labels)}`")

    # 🎯 Calibration Drawing
    if calibrate_mode:
        st.subheader("🎨 Draw bounding boxes for missing labels")
        if zone_img is None:
            st.error("🛑 Zone image is missing or invalid.")
            continue
        if zone_img.mode != "RGB":
            zone_img = zone_img.convert("RGB")
        try:
            canvas_result = st_canvas(
                background_image=zone_img,
                update_streamlit=True,
                height=zone_img.height,
                width=zone_img.width,
                drawing_mode="rect",
                stroke_color="orange",
                key=f"canvas_{i}"
            )
            if canvas_result.json_data and canvas_result.json_data["objects"]:
                for obj in canvas_result.json_data["objects"]:
                    label = obj.get("label") or "UNLABELED"
                    x = obj["left"] / zone_img.width
                    y = obj["top"] / zone_img.height
                    w = obj["width"] / zone_img.width
                    h = obj["height"] / zone_img.height
                    custom_positions[label] = (x, y, w, h)
                st.success(f"✅ Saved {len(canvas_result.json_data['objects'])} boxes for Form {i}")
        except Exception as e:
            st.error(f"📛 Canvas error: {e}")

    # 🟧 Overlay
    learned_avg = average_positions(learned_field_positions)
    if show_overlay and missing_labels:
        zone_img = overlay_missing_labels(zone_img.copy(), missing_labels, learned_avg, fallback_regions, custom_positions)

    st.image(zone_img, caption=f"🧾 Zone {i}", use_container_width=True)

    parsed_forms.append({
        "Form": i,
        "Valid": len(missing_labels) == 0,
        "Missing": missing_labels,
        "Fields": fields
    })

    st.subheader("📋 Extracted Fields")
    st.dataframe(pd.DataFrame(fields), use_container_width=True)

# 📦 Export Data
st.header("💾 Export Data")

# 🧠 Field Flattening
flat_fields = []
for form in parsed_forms:
    flat_fields.extend([{"Form": form["Form"], **field} for field in form["Fields"]])

df_fields = pd.DataFrame(flat_fields)
st.download_button("📄 Download Forms CSV", df_fields.to_csv(index=False), "forms.csv", "text/csv")
st.download_button("📄 Download Forms JSON", json.dumps(flat_fields, indent=2, ensure_ascii=False), "forms.json", "application/json")

# 🧠 Export Custom Calibration Boxes
if custom_positions:
    st.subheader("🧠 Export Calibrated Field Boxes")
    st.json(custom_positions)
    st.download_button(
        label="💾 Download Custom Field Boxes JSON",
        data=json.dumps(custom_positions, indent=2, ensure_ascii=False),
        file_name="custom_field_boxes.json",
        mime="application/json"
    )
else:
    st.info("ℹ️ No custom field boxes drawn yet. Use calibration mode to define them.")

# 🗂️ Summary
st.header("📊 Summary Overview")
valid_forms = [f for f in parsed_forms if f["Valid"]]
invalid_forms = [f for f in parsed_forms if not f["Valid"]]

st.markdown(f"✅ Valid Forms: **{len(valid_forms)}**")
st.markdown(f"❌ Invalid Forms (Missing Fields): **{len(invalid_forms)}**")

if invalid_forms:
    st.subheader("❌ Missing Labels by Form")
    for f in invalid_forms:
        st.markdown(f"- Form {f['Form']}: `{', '.join(f['Missing'])}`")
