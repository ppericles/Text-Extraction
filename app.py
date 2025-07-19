import streamlit as st
from PIL import Image, ImageDraw
import os, json
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 🧠 GCP setup
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
docai_client = documentai.DocumentProcessorServiceClient(
    client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
)

# ✂️ Crop to left half only
def crop_left(image):
    image = image.convert("RGB")
    width, height = image.size
    return image.crop((0, 0, width // 2, height))

# 🧼 Preprocess (Gaussian blur only)
def preprocess(image):
    if image.width > 1500:
        ratio = 1500 / image.width
        image = image.resize((1500, int(image.height * ratio)))
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binarized)

# 🧠 Parse via Document AI
def parse_docai(pil_img):
    name = docai_client.processor_path(project_id, location, processor_id)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    raw_doc = documentai.RawDocument(content=buf.read(), mime_type="image/jpeg")
    req = documentai.ProcessRequest(name=name, raw_document=raw_doc)
    return docai_client.process_document(request=req).document

# 📍 Field grouping + bounding box overlay
def group_fields(document, image, show_boxes=False):
    groups = {1: [], 2: [], 3: []}
    draw = ImageDraw.Draw(image)
    page_height = document.pages[0].dimension.height
    for page in document.pages:
        for field in page.form_fields:
            label = field.field_name.text_anchor.content or ""
            value = field.field_value.text_anchor.content or ""
            confidence = round(field.field_value.confidence * 100, 2)
            verts = field.field_value.bounding_poly.normalized_vertices
            avg_y = sum(v.y for v in verts) / len(verts)
            zone = 1 if avg_y < 0.33 else 2 if avg_y < 0.66 else 3
            groups[zone].append({"Field": label.strip(), "Value": value.strip(), "Confidence": confidence})
            if show_boxes:
                w, h = image.size
                box = [(v.x * w, v.y * h) for v in verts]
                color = "red" if confidence < 50 else "green"
                draw.line(box + [box[0]], fill=color, width=2)
    return groups, image

# ⚙️ Streamlit UI
st.set_page_config(layout="wide", page_title="Greek Registry OCR — Precision Parser")
st.title("📜 Greek Registry OCR — Sharp & Structured")

cred = st.sidebar.file_uploader("🔐 Credentials (.json)", type=["json"])
if cred:
    with open("credentials.json", "wb") as f:
        f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

uploaded = st.file_uploader("📎 Upload registry image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.stop()

show_boxes = st.sidebar.checkbox("🔲 Show bounding boxes (debug)", value=False)

orig = Image.open(uploaded)
left = crop_left(orig)
preproc = preprocess(left)

st.image(orig, caption="📜 Original Full Image", use_column_width=True)
st.image(left, caption="◀️ Cropped Left Half", use_column_width=True)
st.image(preproc, caption="🧼 Preprocessed (Gaussian Blur)", use_column_width=True)

with st.spinner("🔍 Parsing with Document AI..."):
    doc = parse_docai(preproc.copy())
    grouped, overlay = group_fields(doc, preproc.copy(), show_boxes)

if show_boxes:
    st.image(overlay, caption="📦 Bounding Boxes Overlay", use_column_width=True)

all_data = []
form_stats = []

for zone in [1, 2, 3]:
    fields = grouped.get(zone, [])
    st.subheader(f"📄 Φόρμα {zone}")
    if not fields:
        st.info("No fields found.")
        form_stats.append((zone, 0, 0))
        continue
    edited = []
    total_conf = 0
    for i, field in enumerate(fields):
        label, value, conf = field["Field"], field["Value"], field["Confidence"]
        key = f"{zone}_{i}"
        prefix = "🟥 " if conf < 50 else ""
        corrected = st.text_input(f"{prefix}{label} ({conf}%)", value=value, key=key)
        edited.append({"Form": zone, "Field": label, "Value": corrected, "Confidence": conf})
        total_conf += conf
    avg_conf = round(total_conf / len(fields), 2)
    form_stats.append((zone, len(fields), avg_conf))
    st.dataframe(pd.DataFrame(edited), use_container_width=True)
    all_data.extend(edited)

# 📊 Summary
st.subheader("📊 Confidence Summary")
st.dataframe(pd.DataFrame(form_stats, columns=["Form", "Fields Parsed", "Avg Confidence"]), use_container_width=True)

# 💾 Export
st.subheader("💾 Export Results")
df = pd.DataFrame(all_data)
st.download_button("Download CSV", data=df.to_csv(index=False), file_name="form_data.csv", mime="text/csv")
st.download_button("Download JSON", data=json.dumps(all_data, indent=2, ensure_ascii=False), file_name="form_data.json", mime="application/json")
