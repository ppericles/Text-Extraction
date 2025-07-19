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

# ✂️ Crop image to left half
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# 🔍 Document AI client call
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

# 🧬 Extract form fields
def extract_forms(doc):
    zones = {1: [], 2: [], 3: []}
    if not doc or not doc.pages: return zones
    for page in doc.pages:
        for field in page.form_fields:
            label = field.field_name.text_anchor.content or ""
            value = field.field_value.text_anchor.content or ""
            conf = round(field.field_value.confidence * 100, 2)
            box = field.field_value.bounding_poly.normalized_vertices
            avg_y = sum(v.y for v in box) / len(box)
            zone = 1 if avg_y < 0.33 else 2 if avg_y < 0.66 else 3
            zones[zone].append({
                "Label": label.strip(),
                "Raw": value.strip(),
                "Corrected": normalize(value),
                "Confidence": conf,
                "Schema": normalize(label)
            })
    return zones

# 📋 Get paragraphs and header diagnostics
def get_paragraphs(doc, target_headers, threshold=1):
    paragraphs = []
    if not doc or not doc.pages: return paragraphs
    for page in doc.pages:
        for para in page.paragraphs:
            text = para.layout.text_anchor.content or ""
            line_norm = normalize(text)
            if not line_norm: continue
            match_count = sum(1 for h in target_headers if h in line_norm)
            status = "✅ Full Match" if match_count == len(target_headers) else (
                "☑️ Partial Match" if match_count >= threshold else "❌ No Match"
            )
            paragraphs.append({
                "Raw": text.strip(),
                "Normalized": line_norm,
                "Match Status": status,
                "Match Count": match_count
            })
    return paragraphs

# 🧾 Reconstruct table
def extract_table_from_line(all_lines, header_line):
    headers = header_line.split()
    rows = []
    collecting = False
    for line in all_lines:
        if line == header_line:
            collecting = True
            continue
        if collecting:
            parts = line.split()
            row = {headers[i]: parts[i] if i < len(parts) else "" for i in range(len(headers))}
            rows.append(row)
            if len(rows) >= 10: break
    return rows

# 📅 Regex-based date detection
def extract_approximate_dates(text_lines):
    patterns = [
        r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b",
        r"\b\d{1,2}\s+[Α-ΩΪΫ]{3,}\s+\d{2,4}\b"
    ]
    found = set()
    for line in text_lines:
        for pat in patterns:
            matches = re.findall(pat, line)
            found.update(matches)
    return sorted(found)

# 🔠 Greek month → numeric
MONTH_MAP_GR = {
    "ΙΑΝΟΥΑΡΙΟΥ": "01", "ΙΑΝ": "01",
    "ΦΕΒΡΟΥΑΡΙΟΥ": "02", "ΦΕΒ": "02",
    "ΜΑΡΤΙΟΥ": "03", "ΜΑΡ": "03",
    "ΑΠΡΙΛΙΟΥ": "04", "ΑΠΡ": "04",
    "ΜΑΪΟΥ": "05", "ΜΑΙ": "05",
    "ΙΟΥΝΙΟΥ": "06", "ΙΟΥΝ": "06",
    "ΙΟΥΛΙΟΥ": "07", "ΙΟΥΛ": "07",
    "ΑΥΓΟΥΣΤΟΥ": "08", "ΑΥΓ": "08",
    "ΣΕΠΤΕΜΒΡΙΟΥ": "09", "ΣΕΠ": "09",
    "ΟΚΤΩΒΡΙΟΥ": "10", "ΟΚΤ": "10",
    "ΝΟΕΜΒΡΙΟΥ": "11", "ΝΟΕ": "11",
    "ΔΕΚΕΜΒΡΙΟΥ": "12", "ΔΕΚ": "12"
}

def convert_greek_month_dates(text_lines):
    output = []
    for line in text_lines:
        match = re.search(r"(\d{1,2})\s+([Α-ΩΪΫ]{3,})\s+(\d{2,4})", line)
        if match:
            d, m, y = match.groups()
            m_num = MONTH_MAP_GR.get(m.upper())
            if m_num:
                output.append(f"{d.zfill(2)}/{m_num}/{y.zfill(4)}")
    return output

# 🖼 Streamlit UI
st.set_page_config(layout="wide", page_title="Registry OCR Parser")
st.title("🏛️ Registry OCR — Form, Table & Date Extraction")

cred = st.sidebar.file_uploader("🔐 GCP Credentials (.json)", type=["json"])
if cred:
    with open("credentials.json", "wb") as f: f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

file = st.file_uploader("📎 Upload Registry Image", type=["jpg", "jpeg", "png"])
if not file: st.stop()

headers_input = st.sidebar.text_input("📋 Table Headers", value="ΗΜΕΡΟΜΗΝΙΑ,ΕΝΕΡΓΕΙΑ,ΣΧΟΛΙΑ")
target_headers = [normalize(h.strip()) for h in headers_input.split(",") if h.strip()]
threshold = st.sidebar.slider("🔍 Match Threshold", min_value=1, max_value=len(target_headers), value=2)

project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

img = Image.open(file)
left = crop_left(img)
st.image(img, caption="📜 Full Image", use_column_width=True)
st.image(left, caption="🧼 Cropped Left Half", use_column_width=True)

doc = parse_docai(left.copy(), project_id, processor_id, location)
if not doc: st.stop()

# 📄 Display form fields
forms = extract_forms(doc)
st.subheader("📄 Form Field Extraction")
form_stats, all_fields = [], []
for zone in [1,2,3]:
    fields = forms[zone]
    st.markdown(f"### Ζώνη {zone}")
    if not fields: continue
    total = sum(f["Confidence"] for f in fields)
    for i, f in enumerate(fields):
        st.text_input(f"{f['Label']} ({f['Confidence']}%) → [{f['Schema']}]", value=f["Corrected"], key=f"{zone}_{i}")
        all_fields.append(f)
    avg = round(total / len(fields), 2)
    form_stats.append((zone, len(fields), avg))
    st.dataframe(pd.DataFrame(fields), use_container_width=True)

st.subheader("📊 Form Summary")
st.dataframe(pd.DataFrame(form_stats, columns=["Form", "Fields", "Avg Confidence"]), use_container_width=True)

# 🧾 Table diagnostics
paragraphs = get_paragraphs(doc, target_headers, threshold)
ocr_lines = [p["Normalized"] for p in paragraphs]
st.subheader("🔬 OCR Line Diagnostics")
st.dataframe(pd.DataFrame(paragraphs), use_container_width=True)

# 📌 Select header line
st.subheader("📋 Select Header Line for Table Extraction")
valid_lines = [p["Normalized"] for p in paragraphs if p["Match Count"] >= threshold]
selected_header = st.selectbox("Choose Header Line", valid_lines if valid_lines else ["No match found"])

# 🧾 Extract table
st.subheader("🧾 Reconstructed Table")
table_rows = []
if selected_header and selected_header != "No match found":
    table_rows = extract_table_from_line(ocr_lines, selected_header)
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.warning("⚠️ No table rows found under selected header line.")
else:
    st.info("ℹ️ No valid header line selected.")

# 📅 Date Extraction
st.subheader("📅 Date Detection & Standardization")
dates_numeric = extract_approximate_dates(ocr_lines)
dates_greek = convert_greek_month_dates(ocr_lines)
all_dates = sorted(set(dates_numeric + dates_greek))
if all_dates:
    st.dataframe(pd.DataFrame(all_dates, columns=["Detected Dates"]), use_container_width=True)
else:
    st.info("ℹ️ No recognizable dates found.")

# 💾 Export
st.subheader("💾 Export Data")
form_df = pd.DataFrame(all_fields)
st.download_button("📄 Download Forms CSV", form_df.to_csv(index=False), "forms.csv", "text/csv")
st.download_button("📄 Download Forms JSON", json.dumps(all_fields, indent=2, ensure_ascii=False), "forms.json", "application/json")

if table_rows:
    table_df = pd.DataFrame(table_rows)
    st.download_button("🧾 Download Table CSV", table_df.to_csv(index=False), "table.csv", "text/csv")
    st.download_button("🧾 Download Table JSON", json.dumps(table_rows, indent=2, ensure_ascii=False), "table.json", "application/json")
