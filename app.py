import streamlit as st
from PIL import Image
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

# Crop to left half of image
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# Divide image into vertical zones
def split_zones(image):
    w, h = image.size
    bounds = [(0.00, 0.32), (0.33, 0.65), (0.66, 1.00)]
    return [image.crop((0, int(h*t), w, int(h*b))).convert("RGB") for t, b in bounds]

# Document AI parser
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

# Extract fields with label merging and only store if value is present
def extract_fields(doc, target_labels):
    if not doc or not doc.pages: return []

    extracted = {}
    collected = []
    for page in doc.pages:
        for f in page.form_fields:
            label = f.field_name.text_anchor.content or ""
            value = f.field_value.text_anchor.content or ""
            conf = round(f.field_value.confidence * 100, 2)
            collected.append({"Label": label.strip(), "Value": value.strip(), "Confidence": conf})

    i = 0
    while i < len(collected):
        label = collected[i]["Label"]
        value = collected[i]["Value"]
        conf = collected[i]["Confidence"]
        merged_label = label
        merged_value = value
        merged_conf = conf
        found = False

        for j in range(i+1, len(collected)):
            merged_label += " " + collected[j]["Label"]
            merged_value += " " + collected[j]["Value"]
            merged_conf = round((merged_conf + collected[j]["Confidence"]) / 2, 2)

            if merged_value.strip() and any(
                normalize(merged_label).startswith(normalize(t)) or normalize(t) in normalize(merged_label)
                for t in target_labels
            ):
                extracted[merged_label.strip()] = {
                    "Raw": merged_value.strip(), "Corrected": normalize(merged_value),
                    "Confidence": merged_conf, "Schema": normalize(merged_label)
                }
                i = j
                found = True
                break

        if not found and label in target_labels:
            extracted[label] = {
                "Raw": value.strip(), "Corrected": normalize(value),
                "Confidence": conf, "Schema": normalize(label)
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

# 🖼️ Streamlit UI
st.set_page_config(layout="wide", page_title="Greek Registry Parser")
st.title("🏛️ Registry OCR — Validated Extraction")

cred = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred:
    with open("credentials.json", "wb") as f: f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

file = st.file_uploader("📎 Upload Registry Image", type=["jpg", "jpeg", "png"])
if not file:
    st.info("ℹ️ Upload an image to begin")
    st.stop()

img_left = crop_left(Image.open(file))
zones = split_zones(img_left)

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

    # 🔍 Debug toggle — show OCR tokens with confidence
    if st.checkbox(f"🧪 Show raw OCR tokens from Form {i}", value=False):
        tokens = []
        for page in doc.pages:
            for token in page.tokens:
                txt = token.layout.text_anchor.content or ""
                box = token.layout.bounding_poly.normalized_vertices
                y = round(sum(v.y for v in box) / len(box), 3)
                x = round(sum(v.x for v in box) / len(box), 3)
                conf = round(token.layout.confidence * 100, 2)
                tokens.append({"Token": txt, "X": x, "Y": y, "Confidence (%)": conf})
        st.dataframe(pd.DataFrame(tokens), use_container_width=True)

    fields = extract_fields(doc, target_labels)
    table = extract_table(doc)
    dates = convert_greek_month_dates(doc)
    missing = [f["Label"] for f in fields if f["Raw"] == ""]
    valid = len(missing) == 0

    parsed_forms.append({
        "Form": i, "Valid": valid,
        "Missing": missing, "Fields": fields,
        "Table": table, "Dates": dates
    })

    st.subheader("📋 Form Fields")
    st.dataframe(pd.DataFrame(fields), use_container_width=True)
    if not valid:
        st.error(f"❌ Missing: {', '.join(missing)}")

    if table:
        st.subheader("🧾 Table")
        st.dataframe(pd.DataFrame(table), use_container_width=True)
    if dates:
        st.subheader("📅 Dates")
        st.dataframe(pd.DataFrame(dates, columns=["Standardized Date"]), use_container_width=True)

# 📦 Export Section
st.header("💾 Export Data")

flat_fields, flat_tables, flat_dates = [], [], []
for form in parsed_forms:
    flat_fields.extend([
        {"Form": form["Form"], **field} for field in form["Fields"]
    ])
    if form.get("Table"):
        flat_tables.extend([
            {"Form": form["Form"], **row} for row in form["Table"]
        ])
    if form.get("Dates"):
        flat_dates.extend([
            {"Form": form["Form"], "Standardized Date": date} for date in form["Dates"]
        ])

forms_df = pd.DataFrame(flat_fields)
st.download_button("📄 Download Forms CSV", forms_df.to_csv(index=False), "forms.csv", "text/csv")
st.download_button("📄 Download Forms JSON", json.dumps(flat_fields, indent=2, ensure_ascii=False), "forms.json", "application/json")

if flat_tables:
    tables_df = pd.DataFrame(flat_tables)
    st.download_button("🧾 Download Tables CSV", tables_df.to_csv(index=False), "tables.csv", "text/csv")
    st.download_button("🧾 Download Tables JSON", json.dumps(flat_tables, indent=2, ensure_ascii=False), "tables.json", "application/json")

if flat_dates:
    dates_df = pd.DataFrame(flat_dates)
    st.download_button("📅 Download Dates CSV", dates_df.to_csv(index=False), "dates.csv", "text/csv")
    st.download_button("📅 Download Dates JSON", json.dumps(flat_dates, indent=2, ensure_ascii=False), "dates.json", "application/json")
