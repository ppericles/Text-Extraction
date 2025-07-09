import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import difflib

# --- Initialize Google Cloud Vision ---
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = vision.ImageAnnotatorClient(credentials=credentials)

# --- OCR & Extraction Logic ---
def extract_forms_from_ocr(response):
    forms_data = []
    try:
        page = response.full_text_annotation.pages[0]
        img_width = page.width or 1200
        img_height = page.height or 1600
        form_height = img_height / 3
        x_limit = img_width * 0.5

        known_labels = [
            "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
            "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
        ]

        raw_blocks = []
        for block in page.blocks:
            text = ""
            for para in block.paragraphs:
                for word in para.words:
                    text += "".join(symbol.text for symbol in word.symbols) + " "
            text = text.strip()
            if text:
                x = sum(v.x for v in block.bounding_box.vertices) / 4
                y = sum(v.y for v in block.bounding_box.vertices) / 4
                raw_blocks.append({"text": text, "x": x, "y": y})

        forms = {1: [], 2: [], 3: []}
        for b in raw_blocks:
            form_idx = int(b["y"] // form_height) + 1
            if 1 <= form_idx <= 3:
                forms[form_idx].append(b)

        for idx in range(1, 4):
            blocks = forms[idx]
            fields = {label: "" for label in known_labels}
            fields["TABLE_ROWS"] = []

            def match_label(label_hint):
                for b in blocks:
                    if b["x"] > x_limit:
                        continue
                    match = difflib.get_close_matches(label_hint, [b["text"]], cutoff=0.7)
                    if match:
                        return b
                return None

            def find_below(label_hint, x_tol=60, y_min=5, y_max=120):
                label_block = match_label(label_hint)
                if not label_block:
                    return ""
                lx, ly = label_block["x"], label_block["y"]
                candidates = [
                    b for b in blocks
                    if ly + y_min < b["y"] < ly + y_max and abs(b["x"] - lx) < x_tol and b["x"] <= x_limit
                ]
                if candidates:
                    return sorted(candidates, key=lambda b: b["y"])[0]["text"]
                return ""

            for label in known_labels:
                fields[label] = find_below(label)

            fields["TABLE_ROWS"] = [b["text"] for b in blocks if len(b["text"].split()) >= 2][:11]
            forms_data.append(fields)

    except Exception as e:
        forms_data = [{"error": f"❌ OCR extraction failed: {e}"}]

    return forms_data

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Greek OCR Form Parser")
st.title("🧾 Greek Handwriting OCR")

uploaded_file = st.file_uploader("📎 Upload scanned Greek form", type=["png", "jpg", "jpeg"])

if uploaded_file:
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)

    # Optional performance boost: downscale image
    if max(img.size) > 1800:
        img.thumbnail((1800, 1800))

    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Running OCR..."):
        uploaded_file.seek(0)
        img_bytes = uploaded_file.read()

        image_proto = vision.Image(content=img_bytes)
        try:
            response = client.document_text_detection(image=image_proto)
            forms = extract_forms_from_ocr(response)
        except Exception as e:
            st.error(f"❌ Vision API error: {e}")
            st.stop()

    for idx, form in enumerate(forms, start=1):
        with st.expander(f"📄 Φόρμα {idx}", expanded=(idx == 1)):
            if "error" in form:
                st.error(form["error"])
                continue

            r1 = st.columns(5)
            r1[0].text_input("ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", form["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"], key=f"{idx}_0")
            r1[1].text_input("ΕΠΩΝΥΜΟ", form["ΕΠΩΝΥΜΟ"], key=f"{idx}_1")
            r1[2].text_input("ΚΥΡΙΟΝ ΟΝΟΜΑ", form["ΚΥΡΙΟΝ ΟΝΟΜΑ"], key=f"{idx}_2")
            r1[3].text_input("ΟΝΟΜΑ ΠΑΤΡΟΣ", form["ΟΝΟΜΑ ΠΑΤΡΟΣ"], key=f"{idx}_3")
            r1[4].text_input("ΟΝΟΜΑ ΜΗΤΡΟΣ", form["ΟΝΟΜΑ ΜΗΤΡΟΣ"], key=f"{idx}_4")

            r2 = st.columns(3)
            r2[0].text_input("ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", form["ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ"], key=f"{idx}_5")
            r2[1].text_input("ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", form["ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ"], key=f"{idx}_6")
            r2[2].text_input("ΚΑΤΟΙΚΙΑ", form["ΚΑΤΟΙΚΙΑ"], key=f"{idx}_7")

            st.markdown("#### 📋 Πίνακας")
            for i, row in enumerate(form["TABLE_ROWS"]):
                st.text_input(f"Γραμμή {i}", row, key=f"{idx}_table_{i}")
