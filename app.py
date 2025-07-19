import streamlit as st
from PIL import Image
import os, json, unicodedata
import pandas as pd
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 🔠 Normalize Greek: uppercase + remove accents
def normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if unicodedata.category(c) != "Mn")
    return text.upper().strip()

# ✂️ Crop image to left half
def crop_left(image):
    w, h = image.size
    return image.convert("RGB").crop((0, 0, w // 2, h))

# 🔍 Document AI parsing with error feedback
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
        result = client.process_document(request=request)
        return result.document
    except Exception as e:
        st.error(f"📛 Document AI Error: {e}")
        return None

# 🧬 Extract form fields into 3 vertical zones
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

# 📋 Extract OCR lines and diagnose header matches
def get_paragraphs(doc, target_headers, threshold=1):
    paragraphs = []
    if not doc or not doc.pages: return paragraphs
    for page in doc.pages:
        for para in page.paragraphs:
            text = para.layout.text_anchor.content or ""
            line_norm = normalize(text)
            if not line_norm: continue
            match_count = sum(1 for h in target_headers if h in line_norm)
            match_status = "✅ Full Match" if match_count == len(target_headers) else (
                "☑️ Partial Match" if match_count >= threshold else "❌ No Match"
            )
            paragraphs.append({
                "Raw": text.strip(),
                "Normalized": line_norm,
                "Match Status": match_status,
                "Match Count": match_count
            })
    return paragraphs

# 📦 Extract table from selected header line
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

# 🖼 Streamlit UI
st.set_page_config(layout="wide", page_title="Greek Registry Diagnostic OCR")
st.title("🏛️ Registry OCR — Deep Table Diagnostics")

# Sidebar: credentials + header input + match threshold
cred = st.sidebar.file_uploader("🔐 GCP Credentials (.json)", type=["json"])
if cred:
    with open("credentials.json", "wb") as f: f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

file = st.file_uploader("📎 Upload Registry Image", type=["jpg", "jpeg", "png"])
if not file: st.stop()

header_input = st.sidebar.text_input("📋 Table Headers (comma-separated)", value="ΗΜΕΡΟΜΗΝΙΑ,ΕΝΕΡΓΕΙΑ,ΣΧΟΛΙΑ")
target_headers = [normalize(h.strip()) for h in header_input.split(",") if h.strip()]
match_threshold = st.sidebar.slider("🔍 Header Match Threshold", min_value=1, max_value=len(target_headers), value=2)

project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"

# Prepare image
img = Image.open(file)
left = crop_left(img)

st.image(img, caption="📜 Original Image", use_column_width=True)
st.image(left, caption="🧼 Left Half for OCR", use_column_width=True)

doc = parse_docai(left.copy(), project_id, processor_id, location)
if not doc:
    st.warning("🚫 Parsing failed.")
    st.stop()

# 📋 Show form fields
forms = extract_forms(doc)
st.subheader("📄 Form Fields")
form_stats, all_fields = [], []

for zone in [1,2,3]:
    fields = forms[zone]
    st.markdown(f"### Φόρμα {zone}")
    if not fields:
        st.info("No fields found.")
        continue
    total = 0
    for i, f in enumerate(fields):
        st.text_input(f"{f['Label']} ({f['Confidence']}%) → [{f['Schema']}]", value=f["Corrected"], key=f"{zone}_{i}")
        total += f["Confidence"]
        all_fields.append(f)
    avg = round(total / len(fields), 2)
    form_stats.append((zone, len(fields), avg))
    st.dataframe(pd.DataFrame(fields), use_container_width=True)

st.subheader("📊 Form Summary")
st.dataframe(pd.DataFrame(form_stats, columns=["Form", "Fields", "Avg Confidence"]), use_container_width=True)

# 🧾 Diagnose OCR lines
st.subheader("🔬 OCR Line Diagnostics")
paragraphs = get_paragraphs(doc, target_headers, match_threshold)
lines_df = pd.DataFrame(paragraphs)
st.dataframe(lines_df, use_container_width=True)

# 📋 Manual header selection
st.subheader("📌 Select Header Line for Table Extraction")
valid_lines = [p["Normalized"] for p in paragraphs if p["Match Count"] >= match_threshold]
selected_header = st.selectbox("Choose Header Line", valid_lines if valid_lines else ["No match found"])

# 🧾 Extract table
st.subheader("🧾 Extracted Table")
if selected_header and selected_header != "No match found":
    all_lines = [p["Normalized"] for p in paragraphs]
    table = extract_table_from_line(all_lines, selected_header)
    st.dataframe(pd.DataFrame(table), use_container_width=True)

    # Export
    st.subheader("💾 Export Data")
    form_df = pd.DataFrame(all_fields)
    st.download_button("📄 Forms CSV", form_df.to_csv(index=False), "forms.csv", "text/csv")
    st.download_button("📄 Forms JSON", json.dumps(all_fields, indent=2, ensure_ascii=False), "forms.json", "application/json")

    table_df = pd.DataFrame(table)
    st.download_button("🧾 Table CSV", table_df.to_csv(index=False), "table.csv", "text/csv")
    st.download_button("🧾 Table JSON", json.dumps(table, indent=2, ensure_ascii=False), "table.json", "application/json")
else:
    st.warning("⚠️ No valid table header selected. Try increasing threshold or checking OCR quality.")
