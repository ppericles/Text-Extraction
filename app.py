import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os, json, unicodedata
import pandas as pd
import re
from io import BytesIO
from google.cloud import documentai_v1 as documentai
from collections import defaultdict

# Normalize Greek text
def normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFD", text)
    return ''.join(c for c in text if unicodedata.category(c) != "Mn").upper().strip()

# Trim white space
def trim_whitespace(image, intensity_threshold=240, buffer=10):
    gray = image.convert("L")
    pixels = gray.load()
    w, h = gray.size
    top = next((y for y in range(h) if any(pixels[x, y] < intensity_threshold for x in range(w))), 0)
    bottom = next((y for y in reversed(range(h)) if any(pixels[x, y] < intensity_threshold for x in range(w))), h)
    left = next((x for x in range(w) if any(pixels[x, y] < intensity_threshold for y in range(h))), 0)
    right = next((x for x in reversed(range(w)) if any(pixels[x, y] < intensity_threshold for y in range(h))), w)
    top = max(0, top - buffer)
    bottom = min(h, bottom + buffer)
    left = max(0, left - buffer)
    right = min(w, right + buffer)
    return image.crop((left, top, right, bottom))

# Crop to left half
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# Split into fixed zones with overlap
def split_zones_fixed(image, overlap_px):
    w, h = image.size
    thirds = [int(h * t) for t in [0.0, 0.33, 0.66, 1.0]]
    bounds = [
        (thirds[0], thirds[1] + overlap_px),
        (thirds[1] - overlap_px, thirds[2] + overlap_px),
        (thirds[2] - overlap_px, thirds[3])
    ]
    zones = [image.crop((0, max(0, t), w, min(h, b))).convert("RGB") for t, b in bounds]
    return zones, bounds

# Draw zone overlay
def show_zone_overlay(image, bounds, color="red", width=3):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for top, bottom in bounds:
        draw.rectangle([(0, top), (image.width, bottom)], outline=color, width=width)
    return preview

# Overlay missing label boxes
def overlay_missing_labels(zone_img, missing_labels, learned_map, color="orange"):
    draw = ImageDraw.Draw(zone_img)
    w, h = zone_img.size
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except:
        font = None
    for label in missing_labels:
        box = learned_map.get(label)
        if not box: continue
        x, y, bw, bh = box
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + bw) * w), int((y + bh) * h)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        draw.text((x1, y1 - 16), f"Missing: {label}", fill=color, font=font)
    return zone_img

# Parse image with Document AI
def parse_docai(pil_img, project_id, processor_id, location):
    try:
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
        )
        name = client.processor_path(project_id, location, processor_id)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        buf.seek(0)
        raw = documentai.RawDocument(content=buf.read(), mime_type="image/jpeg")
        request = documentai.ProcessRequest(name=name, raw_document=raw)
        return client.process_document(request=request).document
    except Exception as e:
        st.error(f"📛 Document AI Error: {e}")
        return None

# Learn bounding boxes from extracted labels
def learn_field_positions(doc, target_labels):
    positions = defaultdict(list)
    if not doc or not doc.pages: return positions
    for page in doc.pages:
        for f in page.form_fields:
            label = f.field_name.text_anchor.content or ""
            value = f.field_value.text_anchor.content or ""
            box = f.field_value.layout.bounding_poly.normalized_vertices
            if not box or normalize(label) not in [normalize(t) for t in target_labels]: continue
            xs = [v.x for v in box if v]
            ys = [v.y for v in box if v]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            positions[label].append((x, y, w, h))
    return positions

# Compute average region map
def average_positions(position_dict):
    averages = {}
    for label, boxes in position_dict.items():
        if not boxes: continue
        x, y, w, h = zip(*boxes)
        averages[label] = (
            sum(x)/len(x),
            sum(y)/len(y),
            sum(w)/len(w),
            sum(h)/len(h)
        )
    return averages

# Field extraction
def extract_fields(doc, target_labels):
    if not doc or not doc.pages: return []
    extracted = {}
    items = []
    for page in doc.pages:
        for f in page.form_fields:
            label = f.field_name.text_anchor.content or ""
            value = f.field_value.text_anchor.content or ""
            conf = round(f.field_value.confidence * 100, 2)
            items.append({"Label": label.strip(), "Value": value.strip(), "Confidence": conf})
    i = 0
    while i < len(items):
        label = items[i]["Label"]
        value = items[i]["Value"]
        conf = items[i]["Confidence"]
        merged_label = label
        merged_value = value
        merged_conf = conf
        for j in range(i + 1, len(items)):
            merged_label += " " + items[j]["Label"]
            merged_value += " " + items[j]["Value"]
            merged_conf = round((merged_conf + items[j]["Confidence"]) / 2, 2)
            if normalize(merged_label) in [normalize(t) for t in target_labels]:
                extracted[merged_label.strip()] = {
                    "Raw": merged_value.strip(),
                    "Corrected": normalize(merged_value),
                    "Confidence": merged_conf,
                    "Schema": normalize(merged_label)
                }
                i = j
                break
        else:
            if label in target_labels:
                extracted[label] = {
                    "Raw": value.strip(),
                    "Corrected": normalize(value),
                    "Confidence": conf,
                    "Schema": normalize(label)
                }
        i += 1
    fields = []
    for label in target_labels:
        f = extracted.get(label, {
            "Raw": "", "Corrected": "", "Confidence": 0.0, "Schema": normalize(label)
        })
        fields.append({"Label": label, **f})
    return fields
def extract_table(doc):
    tokens = []
    for page in doc.pages:
        for token in page.tokens:
            txt = token.layout.text_anchor.content or ""
            box = token.layout.bounding_poly.normalized_vertices
            y = sum(v.y for v in box) / len(box)
            x = sum(v.x for v in box) / len(box)
            tokens.append({"text": normalize(txt), "y": y, "x": x})
    if not tokens: return []
    tokens.sort(key=lambda t: t["y"])
    rows, current, threshold = [], [], 0.01
    for tok in tokens:
        if not current or abs(tok["y"] - current[-1]["y"]) < threshold:
            current.append(tok)
        else:
            rows.append(current)
            current = [tok]
    if current: rows.append(current)
    if len(rows) < 2: return []
    headers = [t["text"] for t in sorted(rows[0], key=lambda t: t["x"])]
    table = []
    for row in rows[1:]:
        cells = sorted(row, key=lambda t: t["x"])
        table.append({headers[i]: cells[i]["text"] if i < len(cells) else "" for i in range(len(headers))})
    return table

MONTH_MAP_GR = {
    "ΙΑΝΟΥΑΡΙΟΥ": "01", "ΙΑΝ": "01", "ΦΕΒΡΟΥΑΡΙΟΥ": "02", "ΦΕΒ": "02",
    "ΜΑΡΤΙΟΥ": "03", "ΜΑΡ": "03", "ΑΠΡΙΛΙΟΥ": "04", "ΑΠΡ": "04",
    "ΜΑΪΟΥ": "05", "ΜΑΪ": "05", "ΙΟΥΝΙΟΥ": "06", "ΙΟΥΝ": "06",
    "ΙΟΥΛΙΟΥ": "07", "ΙΟΥΛ": "07", "ΑΥΓΟΥΣΤΟΥ": "08", "ΑΥΓ": "08",
    "ΣΕΠΤΕΜΒΡΙΟΥ": "09", "ΣΕΠ": "09", "ΟΚΤΩΒΡΙΟΥ": "10", "ΟΚΤ": "10",
    "ΝΟΕΜΒΡΙΟΥ": "11", "ΝΟΕ": "11", "ΔΕΚΕΜΒΡΙΟΥ": "12", "ΔΕΚ": "12"
}

def convert_greek_month_dates(doc):
    dates = []
    if not doc or not doc.pages: return dates
    for page in doc.pages:
        for token in page.tokens:
            txt = token.layout.text_anchor.content or ""
            match = re.search(r"(\d{1,2})\s+([Α-ΩΪΫ]{3,})\s+(\d{2,4})", normalize(txt))
            if match:
                d, m, y = match.groups()
                m_num = MONTH_MAP_GR.get(m.upper())
                if m_num:
                    dates.append(f"{d.zfill(2)}/{m_num}/{y.zfill(4)}")
    return sorted(set(dates))

# 🖼️ Streamlit UI
st.set_page_config(layout="wide", page_title="Greek Registry Parser")
st.title("🏛️ Registry OCR — Adaptive Missing Field Overlay")

# Sidebar controls
overlap = st.sidebar.slider("🧩 Overlap between form zones", 0, 120, 60, 10)
show_overlay = st.sidebar.checkbox("🟧 Show missing field boxes", value=True)
cred = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred:
    with open("credentials.json", "wb") as f: f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

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

project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
target_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"
]

parsed_forms = []
learned_field_positions = defaultdict(list)

for i, zone_img in enumerate(zones, start=1):
    st.header(f"📄 Form {i}")
    doc = parse_docai(zone_img.copy(), project_id, processor_id, location)
    if not doc: continue
    fields = extract_fields(doc, target_labels)
    table = extract_table(doc)
    dates = convert_greek_month_dates(doc)
    new_positions = learn_field_positions(doc, target_labels)
    for label, boxes in new_positions.items():
        learned_field_positions[label].extend(boxes)
    found_labels = [f["Label"] for f in fields if f["Raw"].strip()]
    missing_labels = [label for label in target_labels if label not in found_labels]
    st.subheader("🕵️ Field Label Report")
    st.markdown(f"✅ Found: `{', '.join(found_labels)}`")
    st.markdown(f"❌ Missing: `{', '.join(missing_labels)}`")
    learned_avg = average_positions(learned_field_positions)
    if show_overlay and missing_labels:
        zone_img = overlay_missing_labels(zone_img.copy(), missing_labels, learned_avg)
    st.image(zone_img, caption=f"🧾 Zone {i}", use_container_width=True)
    parsed_forms.append({
        "Form": i,
        "Valid": len(missing_labels) == 0,
        "Missing": missing_labels,
        "Fields": fields,
        "Table": table,
        "Dates": dates
    })
    st.subheader("📋 Form Fields")
    st.dataframe(pd.DataFrame(fields), use_container_width=True)
    if table:
        st.subheader("🧾 Table")
        st.dataframe(pd.DataFrame(table), use_container_width=True)
    if dates:
        st.subheader("📅 Dates")
        st.dataframe(pd.DataFrame(dates, columns=["Standardized Date"]), use_container_width=True)

# 💾 Export data
st.header("💾 Export Data")
flat_fields, flat_tables, flat_dates = [], [], []
for form in parsed_forms:
    flat_fields.extend([{"Form": form["Form"], **field} for field in form["Fields"]])
    if form.get("Table"):
        flat_tables.extend([{"Form": form["Form"], **row} for row in form["Table"]])
    if form.get("Dates"):
        flat_dates.extend([{"Form": form["Form"], "Standardized Date": date} for date in form["Dates"]])

df_fields = pd.DataFrame(flat_fields)
st.download_button("📄 Download Forms CSV", df_fields.to_csv(index=False), "forms.csv", "text/csv")
st.download_button("📄 Download Forms JSON", json.dumps(flat_fields, indent=2, ensure_ascii=False), "forms.json", "application/json")

if flat_tables:
    df_tables = pd.DataFrame(flat_tables)
    st.download_button("🧾 Download Tables CSV", df_tables.to_csv(index=False), "tables.csv", "text/csv")
    st.download_button("🧾 Download Tables JSON", json.dumps(flat_tables, indent=2, ensure_ascii=False), "tables.json", "application/json")

if flat_dates:
    df_dates = pd.DataFrame(flat_dates)
    st.download_button("📅 Download Dates CSV", df_dates.to_csv(index=False), "dates.csv", "text/csv")
    st.download_button("📅 Download Dates JSON", json.dumps(flat_dates, indent=2, ensure_ascii=False), "dates.json", "application/json")
