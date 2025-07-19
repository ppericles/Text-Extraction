import streamlit as st
from PIL import Image
import os, json, unicodedata
import pandas as pd
import re
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 🔠 Normalize Greek text
def normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFD", text)
    return ''.join(c for c in text if unicodedata.category(c) != "Mn").upper().strip()

# ✂️ Crop to left half
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# 📐 Tuned vertical slicing (manually defined)
def split_zones_tuned(image):
    w, h = image.size
    boundaries = [(0.00, 0.32), (0.33, 0.65), (0.66, 1.00)]  # Adjust as needed
    zones = []
    for top_pct, bottom_pct in boundaries:
        top = int(h * top_pct)
        bottom = int(h * bottom_pct)
        zones.append(image.crop((0, top, w, bottom)).convert("RGB"))
    return zones

# 🔍 Document AI parsing
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

# 📋 Form fields
def extract_fields(doc):
    fields = []
    if not doc or not doc.pages: return fields
    for page in doc.pages:
        for f in page.form_fields:
            label = f.field_name.text_anchor.content or ""
            value = f.field_value.text_anchor.content or ""
            conf = round(f.field_value.confidence * 100, 2)
            fields.append({
                "Label": label.strip(),
                "Raw": value.strip(),
                "Corrected": normalize(value),
                "Confidence": conf,
                "Schema": normalize(label)
            })
    return fields

# 📊 Table extraction
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

    header_cells = sorted(rows[0], key=lambda t: t["x"])
    headers = [t["text"] for t in header_cells]
    table = []
    for row in rows[1:11]:
        cells = sorted(row, key=lambda t: t["x"])
        table.append({headers[i]: cells[i]["text"] if i < len(cells) else "" for i in range(len(headers))})
    return table

# 📅 Greek month → number map
MONTH_MAP_GR = {
    "ΙΑΝΟΥΑΡΙΟΣ": "01", "ΙΑΝΟΥΑΡΙΟΥ": "01", "ΙΑΝ": "01",
    "ΦΕΒΡΟΥΑΡΙΟΣ": "02", "ΦΕΒΡΟΥΑΡΙΟΥ": "02", "ΦΕΒ": "02",
    "ΜΑΡΤΙΟΣ": "03", "ΜΑΡΤΙΟΥ": "03", "ΜΑΡ": "03",
    "ΑΠΡΙΛΙΟΣ": "04", "ΑΠΡΙΛΙΟΥ": "04", "ΑΠΡ": "04",
    "ΜΑΙΟΣ": "05", "ΜΑΪΟΥ": "05", "ΜΑΪ": "05",
    "ΙΟΥΝΙΟΣ": "06", "ΙΟΥΝΙΟΥ": "06", "ΙΟΥΝ": "06",
    "ΙΟΥΛΙΟΣ": "07", "ΙΟΥΛΙΟΥ": "07", "ΙΟΥΛ": "07",
    "ΑΥΓΟΥΣΤΟΣ": "08", "ΑΥΓΟΥΣΤΟΥ": "08", "ΑΥΓ": "08",
    "ΣΕΠΤΕΜΒΡΙΟΣ": "09", "ΣΕΠΤΕΜΒΡΙΟΥ": "09", "ΣΕΠ": "09",
    "ΟΚΤΩΒΡΙΟΣ": "10", "ΟΚΤΩΒΡΙΟΥ": "10", "ΟΚΤ": "10",
    "ΝΟΕΜΒΡΙΟΣ": "11", "ΝΟΕΜΒΡΙΟΥ": "11", "ΝΟΕ": "11",
    "ΔΕΚΕΜΒΡΙΟΣ": "12", "ΔΕΚΕΜΒΡΙΟΥ": "12", "ΔΕΚ": "12"
}

def convert_greek_month_dates(doc):
    dates = []
    if not doc or not doc.pages: return dates
    for page in doc.pages:
        for token in page.tokens:
            text = token.layout.text_anchor.content or ""
            match = re.search(r"(\d{1,2})\s+([Α-ΩΪΫ]{3,})\s+(\d{2,4})", normalize(text))
            if match:
                day, month_name, year = match.groups()
                month_num = MONTH_MAP_GR.get(month_name.upper())
                if month_num:
                    dates.append(f"{day.zfill(2)}/{month_num}/{year.zfill(4)}")
    return sorted(set(dates))

# 🖼️ Streamlit UI
st.set_page_config(layout="wide", page_title="Greek Registry Parser")
st.title("🏛️ Registry OCR — Left Half (3 Calibrated Zones)")

cred = st.sidebar.file_uploader("🔐 GCP Credentials", type=["json"])
if cred:
    with open("credentials.json", "wb") as f: f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

file = st.file_uploader("📎 Upload Registry Image", type=["jpg", "jpeg", "png"])
if not file: st.stop()

img = Image.open(file)
left_img = crop_left(img)
zones = split_zones_tuned(left_img)

project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

all_fields, all_tables, all_dates = [], [], []

for i, zone_img in enumerate(zones, start=1):
    st.header(f"📄 Form {i}")
    st.image(zone_img, caption=f"🧾 Zone {i} (Left Half)", use_column_width=True)
    doc = parse_docai(zone_img.copy(), project_id, processor_id, location)
    if not doc: continue

    fields = extract_fields(doc)
    table = extract_table(doc)
    dates = convert_greek_month_dates(doc)

    if fields:
        all_fields.extend(fields)
        st.subheader("📋 Form Fields")
        st.dataframe(pd.DataFrame(fields), use_container_width=True)
    else:
        st.warning("⚠️ No form fields detected.")

    if table:
        all_tables.extend(table)
        st.subheader("🧾 Table")
        st.dataframe(pd.DataFrame(table), use_container_width=True)
    else:
        st.warning("⚠️ No table data detected.")

    if dates:
        all_dates.extend(dates)
        st.subheader("📅 Greek Date Conversion")
        st.dataframe(pd.DataFrame(dates, columns=["Standardized Date"]), use_container_width=True)
    else:
        st.info("ℹ️ No Greek-style dates found.")

# 💾 Export Section
st.header("💾 Export Combined Data")

forms_df = pd.DataFrame(all_fields)
st.download_button("📄 Download Forms CSV", forms_df.to_csv(index=False), "forms.csv", "text/csv")
st.download_button("📄 Download Forms JSON", json.dumps(all_fields, indent=2, ensure_ascii=False), "forms.json", "application/json")

if all_tables:
    tables_df = pd.DataFrame(all_tables)
    st.download_button("🧾 Download Tables CSV", tables_df.to_csv(index=False), "tables.csv", "text/csv")
    st.download_button("🧾 Download Tables JSON", json.dumps(all_tables, indent=2, ensure_ascii=False), "tables.json", "application/json")

if all_dates:
    dates_df = pd.DataFrame(sorted(set(all_dates)), columns=["Date"])
    st.download_button("📅 Download Greek Dates CSV", dates_df.to_csv(index=False), "dates.csv", "text/csv")
    st.download_button("📅 Download Greek Dates JSON", json.dumps(sorted(set(all_dates)), indent=2, ensure_ascii=False), "dates.json", "application/json")
