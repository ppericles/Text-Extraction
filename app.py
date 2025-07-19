import streamlit as st
from PIL import Image, ImageDraw
import os, json, unicodedata
import pandas as pd
import re
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# Normalize Greek text
def normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFD", text)
    return ''.join(c for c in text if unicodedata.category(c) != "Mn").upper().strip()

# Trim white space around content
def trim_whitespace(image, threshold=245):
    img_gray = image.convert("L")
    bbox = img_gray.point(lambda p: p < threshold and 255).getbbox()
    return image.crop(bbox) if bbox else image

# Crop image to left half
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# Slice fixed vertical zones with overlap
def split_zones_fixed(image, overlap_px):
    w, h = image.size
    thirds = [int(h * t) for t in [0.0, 0.33, 0.66, 1.0]]
    bounds = [
        (thirds[0], thirds[1] + overlap_px),
        (thirds[1] - overlap_px, thirds[2] + overlap_px),
        (thirds[2] - overlap_px, thirds[3])
    ]
    return [image.crop((0, max(0, t), w, min(h, b))).convert("RGB") for t, b in bounds], bounds

# Auto-detect horizontal gaps to segment forms
def split_zones_auto(image, min_gap_height=20):
    img_gray = image.convert("L")
    histogram = [sum(img_gray.getpixel((x, y)) < 240 for x in range(image.width)) for y in range(image.height)]
    gaps = [i for i in range(1, len(histogram)) if histogram[i] < 5 and histogram[i-1] >= 5]
    chunk_bounds = []
    if not gaps: return [image.copy()], [(0, image.height)]

    start = 0
    for gap in gaps:
        if gap - start > min_gap_height:
            chunk_bounds.append((start, gap))
            start = gap + 1
    if start < image.height:
        chunk_bounds.append((start, image.height))

    zones = [image.crop((0, t, image.width, b)).convert("RGB") for t, b in chunk_bounds]
    return zones, chunk_bounds

# Visual preview of zone boundaries
def show_zone_overlay(image, bounds, color="red", width=4):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for top, bottom in bounds:
        draw.rectangle([(0, top), (image.width, bottom)], outline=color, width=width)
    return preview

# Send image to Document AI
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

# Extract fields from form
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
    if len(rows) < 11: return []

    headers = [t["text"] for t in sorted(rows[0], key=lambda t: t["x"])]
    table = []
    for row in rows[1:11]:
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

# Streamlit UI
st.set_page_config(layout="wide", page_title="Greek Registry Parser")
st.title("🏛️ Registry OCR — Interactive Zone Extraction")

# Sidebar controls
overlap = st.sidebar.slider("🧩 Overlap between form zones (pixels)", 0, 120, value=60, step=10)
auto_mode = st.sidebar.checkbox("🔍 Use auto-detected zones", value=False)
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

if auto_mode:
    zones, bounds = split_zones_auto(img_left)
else:
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

for i, zone_img in enumerate(zones, start=1):
    st.header(f"📄 Form {i}")
    st.image(zone_img, caption=f"🧾 Zone {i}", use_container_width=True)
    doc = parse_docai(zone_img.copy(), project_id, processor_id, location)
    if not doc: continue

    fields = extract_fields(doc, target_labels)
    table = extract_table(doc)
    dates = convert_greek_month_dates(doc)

    found_labels = [f["Label"] for f in fields if f["Raw"].strip()]
    missing_labels = [label for label in target_labels if label not in found_labels]

    st.subheader("🕵️ Field Label Report")
    st.markdown(f"✅ Found: `{', '.join(found_labels)}`")
    st.markdown(f"❌ Missing: `{', '.join(missing_labels)}`")

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

# Export section
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
