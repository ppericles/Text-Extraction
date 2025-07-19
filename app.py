import streamlit as st
from PIL import Image
import os, json, unicodedata
import pandas as pd
import re
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# Text normalization
def normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFD", text)
    return ''.join(c for c in text if unicodedata.category(c) != "Mn").upper().strip()

# Crop image to left half
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# Split into 3 vertical zones
def split_zones(image):
    w, h = image.size
    zones = [(0.00, 0.32), (0.33, 0.65), (0.66, 1.00)]
    return [image.crop((0, int(h*t), w, int(h*b))).convert("RGB") for t, b in zones]

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

# Extract fields with flexible label merging
def extract_fields(doc, target_labels):
    if not doc or not doc.pages: return []

    extracted = {}
    raw_list = []
    for page in doc.pages:
        for f in page.form_fields:
            label = f.field_name.text_anchor.content or ""
            value = f.field_value.text_anchor.content or ""
            conf = round(f.field_value.confidence * 100, 2)
            raw_list.append({"Label": label.strip(), "Value": value.strip(), "Confidence": conf})

    # Merge labels
    i = 0
    while i < len(raw_list):
        lbl = raw_list[i]["Label"]
        val = raw_list[i]["Value"]
        conf = raw_list[i]["Confidence"]
        merged_lbl = lbl
        merged_val = val
        merged_conf = conf
        found = False

        for j in range(i+1, len(raw_list)):
            merged_lbl += " " + raw_list[j]["Label"]
            merged_val += " " + raw_list[j]["Value"]
            merged_conf = round((merged_conf + raw_list[j]["Confidence"]) / 2, 2)
            if any(normalize(merged_lbl).startswith(normalize(t)) or normalize(t) in normalize(merged_lbl) for t in target_labels):
                extracted[merged_lbl.strip()] = {
                    "Raw": merged_val.strip(), "Corrected": normalize(merged_val),
                    "Confidence": merged_conf, "Schema": normalize(merged_lbl)
                }
                i = j
                found = True
                break

        if not found and lbl in target_labels:
            extracted[lbl] = {
                "Raw": val.strip(), "Corrected": normalize(val),
                "Confidence": conf, "Schema": normalize(lbl)
            }
        i += 1

    # Fill missing keys
    fields = []
    for label in target_labels:
        data = extracted.get(label, {
            "Raw": "", "Corrected": "", "Confidence": 0.0, "Schema": normalize(label)
        })
        fields.append({"Label": label, **data})
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

# Greek month detection
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

# UI and export
st.header("💾 Export Data")

flat_fields, flat_tables, flat_dates = [], [], []

for form in parsed_forms:
    flat_fields.extend([
        {"Form": form["Form"], **field}
        for field in form["Fields"]
    ])
    if form.get("Table"):
        flat_tables.extend([
            {"Form": form["Form"], **row}
            for row in form["Table"]
        ])
    if form.get("Dates"):
        flat_dates.extend([
            {"Form": form["Form"], "Standardized Date": date}
            for date in form["Dates"]
        ])

fields_df = pd.DataFrame(flat_fields)
st.download_button("📄 Download Forms CSV", fields_df.to_csv(index=False), "forms.csv", "text/csv")
st.download_button("📄 Download Forms JSON", json.dumps(flat_fields, indent=2, ensure_ascii=False), "forms.json", "application/json")

if flat_tables:
    tables_df = pd.DataFrame(flat_tables)
    st.download_button("🧾 Download Tables CSV", tables_df.to_csv(index=False), "tables.csv", "text/csv")
    st.download_button("🧾 Download Tables JSON", json.dumps(flat_tables, indent=2, ensure_ascii=False), "tables.json", "application/json")

if flat_dates:
    dates_df = pd.DataFrame(flat_dates)
    st.download_button("📅 Download Dates CSV", dates_df.to_csv(index=False), "dates.csv", "text/csv")
    st.download_button("📅 Download Dates JSON", json.dumps(flat_dates, indent=2, ensure_ascii=False), "dates.json", "application/json")
