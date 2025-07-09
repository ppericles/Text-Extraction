import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import difflib

# --- Initialize Vision API Client ---
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = vision.ImageAnnotatorClient(credentials=credentials)

# --- Form OCR Extraction Function ---
def extract_forms_from_ocr(response):
    forms_data = []

    try:
        page = response.full_text_annotation.pages[0]
        img_width = page.width or 1200
        img_height = page.height or 1600
        form_height = img_height / 3
        x_limit = img_width * 0.5  # Only left half used for headers

        known_labels = [
            "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
            "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
        ]

        # Collect all OCR blocks with position
        raw_blocks = []
        for block in page.blocks:
            text = "".join(
                symbol.text
                for para in block.paragraphs
                for word in para.words
                for symbol in word.symbols
            ).strip()
            if text:
                x = sum(v.x for v in block.bounding_box.vertices) / 4
                y = sum(v.y for v in block.bounding_box.vertices) / 4
                raw_blocks.append({"text": text, "x": x, "y": y})

        # Group by vertical zone into 3 forms
        forms = {1: [], 2: [], 3: []}
        for b in raw_blocks:
            idx = int(b["y"] // form_height) + 1
            if 1 <= idx <= 3:
                forms[idx].append(b)

        # Extraction helpers
        def match_label(label_hint, blocks):
            for b in blocks:
                if b["x"] <= x_limit:
                    match = difflib.get_close_matches(label_hint, [b["text"]], cutoff=0.6)
                    if match:
                        return b
            return None

        def find_value_below(label_text, blocks, x_tol=80, y_min=5, y_max=150):
            label = match_label(label_text, blocks)
            if not label:
                return "", 0
            lx, ly = label["x"], label["y"]
            candidates = [
                b for b in blocks
                if ly + y_min < b["y"] < ly + y_max and abs(b["x"] - lx) < x_tol and b["x"] <= x_limit
            ]
            if candidates:
                best = sorted(candidates, key=lambda b: b["y"])[0]
                return best["text"], best["y"]
            return "", ly

        # Extract fields for each form
        for i in range(1, 4):
            blocks = forms[i]
            fields = {}
            max_header_y = 0

            for label in known_labels:
                val, val_y = find_value_below(label, blocks)
                fields[label] = val
                if val_y > max_header_y:
                    max_header_y = val_y

            # Extract table rows below max_header_y and not far left-aligned
            table_candidates = [
                b["text"] for b in blocks
                if b["y"] > max_header_y + 20 and b["x"] > x_limit * 0.8 and len(b["text"].split()) >= 2
            ]
            fields["TABLE_ROWS"] = table_candidates[:11]
            forms_data.append(fields)

    except Exception as e:
        forms_data = [{"error": f"❌ OCR parsing failed: {e}"}]

    return forms_data

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Greek OCR Form Parser")
st.title("📄 Greek Handwritten OCR Form Extractor")

uploaded_file = st.file_uploader("📎 Upload a Greek form image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)

    if max(img.size) > 1800:
        img.thumbnail((1800, 1800))

    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Processing with Google Vision..."):
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

            c1 = st.columns(5)
            c1[0].text_input("ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", form.get("ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", ""), key=f"{idx}_0")
            c1[1].text_input("ΕΠΩΝΥΜΟ", form.get("ΕΠΩΝΥΜΟ", ""), key=f"{idx}_1")
            c1[2].text_input("ΚΥΡΙΟΝ ΟΝΟΜΑ", form.get("ΚΥΡΙΟΝ ΟΝΟΜΑ", ""), key=f"{idx}_2")
            c1[3].text_input("ΟΝΟΜΑ ΠΑΤΡΟΣ", form.get("ΟΝΟΜΑ ΠΑΤΡΟΣ", ""), key=f"{idx}_3")
            c1[4].text_input("ΟΝΟΜΑ ΜΗΤΡΟΣ", form.get("ΟΝΟΜΑ ΜΗΤΡΟΣ", ""), key=f"{idx}_4")

            c2 = st.columns(3)
            c2[0].text_input("ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", form.get("ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", ""), key=f"{idx}_5")
            c2[1].text_input("ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", form.get("ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", ""), key=f"{idx}_6")
            c2[2].text_input("ΚΑΤΟΙΚΙΑ", form.get("ΚΑΤΟΙΚΙΑ", ""), key=f"{idx}_7")

            st.markdown("#### 📋 Πίνακας")
            for i, row in enumerate(form.get("TABLE_ROWS", [])):
                st.text_input(f"Γραμμή {i}", row, key=f"{idx}_table_{i}")
