import streamlit as st
from PIL import Image, ImageDraw
import os, json
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from google.cloud import documentai_v1 as documentai

# 🧠 GCP config
project_id = "heroic-gantry-380919"
processor_id = "8f7f56e900fbb37e"
location = "eu"
docai_client = documentai.DocumentProcessorServiceClient(
    client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
)

# ✂️ Crop to left half
def crop_left(image):
    image = image.convert("RGB")
    w, h = image.size
    return image.crop((0, 0, w // 2, h))

# 🧼 Preprocessing
def preprocess(image, denoise=True):
    if image.width > 1500:
        image = image.resize((1500, int(image.height * 1500 / image.width)))
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21) if denoise else cv2.GaussianBlur(gray, (3, 3), 0)
    _, binarized = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binarized)

# 🧠 Call Document AI
def parse_docai(pil_img):
    name = docai_client.processor_path(project_id, location, processor_id)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    raw_doc = documentai.RawDocument(content=buf.read(), mime_type="image/jpeg")
    req = documentai.ProcessRequest(name=name, raw_document=raw_doc)
    return docai_client.process_document(request=req).document

# 🧾 Group fields + visualize bounding boxes
def group_fields(document, image, show_boxes=False):
    zones = {1: [], 2: [], 3: []}
    draw = ImageDraw.Draw(image)
    h = document.pages[0].dimension.height
    for page in document.pages:
        for field in page.form_fields:
            key = field.field_name.text_anchor.content or ""
            val = field.field_value.text_anchor.content or ""
            conf = round(field.field_value.confidence * 100, 2)
            verts = field.field_value.bounding_poly.normalized_vertices
            avg_y = sum(v.y for v in verts) / len(verts)
            zone = 1 if avg_y < 0.33 else 2 if avg_y < 0.66 else 3
            zones[zone].append({"Field": key.strip(), "Value": val.strip(), "Confidence": conf})
            if show_boxes:
                w, h_img = image.size
                box = [(v.x * w, v.y * h_img) for v in verts]
                color = "red" if conf < 50 else "green"
                draw.line(box + [box[0]], fill=color, width=2)
    return zones, image

# 🔧 Streamlit UI
st.set_page_config(layout="wide", page_title="Greek Registry OCR — Smart Extract")
st.title("🧭 Greek Registry OCR — Left Half + Bounding Boxes")

cred = st.sidebar.file_uploader("🔐 Credentials (.json)", type=["json"])
if cred:
    with open("credentials.json", "wb") as f:
        f.write(cred.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

uploaded = st.file_uploader("📎 Upload registry image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.stop()

denoise = st.sidebar.checkbox("🧼 Apply denoising", value=True)
show_boxes = st.sidebar.checkbox("🔲 Show bounding boxes (debug)", value=False)

original = Image.open(uploaded)
left_half = crop_left(original)
processed = preprocess(left_half, denoise=denoise)

st.image(original, caption="📜 Full Image", use_column_width=True)
st.image(left_half, caption="◀️ Left Half", use_column_width=True)
st.image(processed, caption="🧼 Preprocessed", use_column_width=True)

with st.spinner("🔍 Parsing with Document AI..."):
    doc = parse_docai(processed.copy())
    grouped, boxed = group_fields(doc, processed.copy(), show_boxes)

if show_boxes:
    st.image(boxed, caption="📦 Bounding Boxes Overlay", use_column_width=True)

# 📊 Display forms
all_fields = []
form_stats = []

for zone in [1, 2, 3]:
    fields = grouped.get(zone, [])
    st.subheader(f"📄 Φόρμα {zone}")
    if not fields:
        st.info("No fields found in this zone.")
        form_stats.append((zone, 0, 0))
        continue
    edited = []
    total = 0
    for i, field in enumerate(fields):
        label, val, conf = field["Field"], field["Value"], field["Confidence"]
        key = f"{zone}_{i}"
        color = "🟥" if conf < 50 else ""
        corrected = st.text_input(f"{color}{label} ({conf}%)", value=val, key=key)
        edited.append({"Form": zone, "Field": label, "Value": corrected, "Confidence": conf})
        total += conf
    form_stats.append((zone, len(fields), round(total / len(fields), 2)))
    st.dataframe(pd.DataFrame(edited), use_container_width=True)
    all_fields.extend(edited)

# 📊 Confidence dashboard
st.subheader("📊 Form Extraction Summary")
st.dataframe(pd.DataFrame(form_stats, columns=["Form", "Fields", "Avg Confidence"]), use_container_width=True)

# 💾 Export
st.subheader("💾 Export Results")
df = pd.DataFrame(all_fields)
st.download_button("Download CSV", df.to_csv(index=False), "parsed_forms.csv", "text/csv")
st.download_button("Download JSON", json.dumps(all_fields, indent=2, ensure_ascii=False), "parsed_forms.json", "application/json")
