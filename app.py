import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account

# --- Initialize Vision API client ---
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = vision.ImageAnnotatorClient(credentials=credentials)

# --- OCR Form Parser with Spatial Rules ---
def extract_forms_from_ocr(response):
    forms_data = []

    try:
        page = response.full_text_annotation.pages[0]
        img_width = page.width or 1200
        img_height = page.height or 1600
        form_height = img_height / 3
        left_x_limit = img_width * 0.5  # Restrict field detection to left half

        # Get all block texts with position
        raw_blocks = []
        for block in page.blocks:
            text = ""
            for para in block.paragraphs:
                for word in para.words:
                    text += "".join(s.text for s in word.symbols) + " "
            text = text.strip()
            if text:
                x = sum(v.x for v in block.bounding_box.vertices) / 4
                y = sum(v.y for v in block.bounding_box.vertices) / 4
                raw_blocks.append({"text": text, "x": x, "y": y})

        # Divide into 3 forms by vertical region
        forms = {1: [], 2: [], 3: []}
        for b in raw_blocks:
            form_idx = int(b["y"] // form_height) + 1
            if 1 <= form_idx <= 3:
                forms[form_idx].append(b)

        # Helper: find value below a label in left half
        def find_below(form_blocks, label_text, x_limit=600, x_tol=60, y_min=5, y_max=150):
            label = next((b for b in form_blocks if label_text in b["text"] and b["x"] <= x_limit), None)
            if not label:
                return ""
            lx, ly = label["x"], label["y"]
            candidates = [
                b for b in form_blocks
                if ly + y_min < b["y"] < ly + y_max and abs(b["x"] - lx) < x_tol and b["x"] <= x_limit
            ]
            if candidates:
                return sorted(candidates, key=lambda b: b["y"])[0]["text"]
            return ""

        # Parse each form
        for idx in range(1, 4):
            blocks = forms[idx]
            fields = {
                "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": find_below(blocks, "ΜΕΡΙΔΟΣ"),
                "ΕΠΩΝΥΜΟ": find_below(blocks, "ΕΠΩΝΥΜΟ"),
                "ΚΥΡΙΟΝ ΟΝΟΜΑ": find_below(blocks, "ΚΥΡΙΟΝ"),
                "ΟΝΟΜΑ ΠΑΤΡΟΣ": find_below(blocks, "ΠΑΤΡΟΣ"),
                "ΟΝΟΜΑ ΜΗΤΡΟΣ": find_below(blocks, "ΜΗΤΡΟΣ"),
                "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ": find_below(blocks, "ΓΕΝΝΗΣΕΩΣ"),
                "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ": find_below(blocks, "ΕΤΟΣ"),
                "ΚΑΤΟΙΚΙΑ": find_below(blocks, "ΚΑΤΟΙΚΙΑ"),
                "TABLE_ROWS": [b["text"] for b in blocks if len(b["text"].split()) >= 2][:11]
            }
            forms_data.append(fields)

    except Exception as e:
        forms_data = [{"error": f"❌ OCR parse error: {e}"}]

    return forms_data

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Greek Form OCR")
st.title("📄 Greek Handwriting OCR – Multi-Form Parser")

uploaded_file = st.file_uploader("📎 Upload handwritten Greek form image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Extracting form data..."):
        uploaded_file.seek(0)
        img_bytes = uploaded_file.read()
        if not img_bytes:
            st.error("⚠️ Image file is empty.")
            st.stop()

        image_proto = vision.Image(content=img_bytes)
        try:
            response = client.document_text_detection(image=image_proto)
            forms = extract_forms_from_ocr(response)
        except Exception as e:
            st.error(f"❌ Vision API error: {e}")
            st.stop()

    for idx, form in enumerate(forms, start=1):
        st.markdown(f"---\n## 📄 Φόρμα {idx}")
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
