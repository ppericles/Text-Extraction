import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account

# --- GCP Vision Client ---
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = vision.ImageAnnotatorClient(credentials=credentials)

# --- OCR Parsing Helper ---
def extract_forms_from_ocr(response):
    forms_data = []

    try:
        page = response.full_text_annotation.pages[0]
        image_height = page.height or 1000
        form_height = image_height / 3

        # Step 1: Extract all block texts with center positions
        raw_blocks = []
        for block in page.blocks:
            text = ""
            for para in block.paragraphs:
                for word in para.words:
                    text += ''.join(symbol.text for symbol in word.symbols) + " "
            text = text.strip()
            if text:
                x_center = sum(v.x for v in block.bounding_box.vertices) / 4
                y_center = sum(v.y for v in block.bounding_box.vertices) / 4
                raw_blocks.append({"text": text, "x": x_center, "y": y_center})

        # Step 2: Group blocks into 3 vertical form zones
        forms = {1: [], 2: [], 3: []}
        for b in raw_blocks:
            form_idx = int(b["y"] // form_height) + 1
            if 1 <= form_idx <= 3:
                forms[form_idx].append(b)

        # Step 3: Extract fields per form
        for idx in range(1, 4):
            form_blocks = forms[idx]
            fields = {
                "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": "", "ΕΠΩΝΥΜΟ": "", "ΚΥΡΙΟΝ ΟΝΟΜΑ": "",
                "ΟΝΟΜΑ ΠΑΤΡΟΣ": "", "ΟΝΟΜΑ ΜΗΤΡΟΣ": "",
                "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ": "", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ": "", "ΚΑΤΟΙΚΙΑ": "",
                "TABLE_ROWS": []
            }

            def find_below(label_text, x_tolerance=60, y_min=5, y_max=150):
                label = next((b for b in form_blocks if label_text in b["text"]), None)
                if not label:
                    return ""
                lx, ly = label["x"], label["y"]
                candidates = [
                    b for b in form_blocks
                    if ly + y_min < b["y"] < ly + y_max and abs(b["x"] - lx) < x_tolerance
                ]
                if candidates:
                    return sorted(candidates, key=lambda b: b["y"])[0]["text"]
                return ""

            # Extract fields based on vertical proximity
            fields["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"] = find_below("ΜΕΡΙΔΟΣ")
            fields["ΕΠΩΝΥΜΟ"] = find_below("ΕΠΩΝΥΜΟ")
            fields["ΚΥΡΙΟΝ ΟΝΟΜΑ"] = find_below("ΚΥΡΙΟΝ")
            fields["ΟΝΟΜΑ ΠΑΤΡΟΣ"] = find_below("ΠΑΤΡΟΣ")
            fields["ΟΝΟΜΑ ΜΗΤΡΟΣ"] = find_below("ΜΗΤΡΟΣ")
            fields["ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ"] = find_below("ΓΕΝΝΗΣΕΩΣ")
            fields["ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ"] = find_below("ΕΤΟΣ")
            fields["ΚΑΤΟΙΚΙΑ"] = find_below("ΚΑΤΟΙΚΙΑ")

            # Heuristically extract long blocks as table rows
            table_rows = [b["text"] for b in form_blocks if len(b["text"].split()) >= 2]
            fields["TABLE_ROWS"] = table_rows[:11]

            forms_data.append(fields)

    except Exception as e:
        forms_data = [{"error": f"❌ OCR extraction failed: {e}"}]

    return forms_data

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Greek Form OCR")
st.title("📄 Greek Handwriting OCR – Form Parser")

uploaded_file = st.file_uploader("📎 Upload a handwritten Greek form image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Extracting form data..."):
        uploaded_file.seek(0)
        content = uploaded_file.read()
        if not content:
            st.error("⚠️ File is empty.")
            st.stop()

        image_proto = vision.Image(content=content)
        try:
            response = client.document_text_detection(image=image_proto)
        except Exception as e:
            st.error(f"❌ Google Vision error: {e}")
            st.stop()

        forms = extract_forms_from_ocr(response)

    for idx, form in enumerate(forms, start=1):
        st.markdown(f"---\n## 📄 Φόρμα {idx}")
        if "error" in form:
            st.error(form["error"])
            continue

        # Row 1 – 5 columns
        r1 = st.columns(5)
        r1[0].text_input("ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", form["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"], key=f"{idx}_0")
        r1[1].text_input("ΕΠΩΝΥΜΟ", form["ΕΠΩΝΥΜΟ"], key=f"{idx}_1")
        r1[2].text_input("ΚΥΡΙΟΝ ΟΝΟΜΑ", form["ΚΥΡΙΟΝ ΟΝΟΜΑ"], key=f"{idx}_2")
        r1[3].text_input("ΟΝΟΜΑ ΠΑΤΡΟΣ", form["ΟΝΟΜΑ ΠΑΤΡΟΣ"], key=f"{idx}_3")
        r1[4].text_input("ΟΝΟΜΑ ΜΗΤΡΟΣ", form["ΟΝΟΜΑ ΜΗΤΡΟΣ"], key=f"{idx}_4")

        # Row 2 – 3 columns
        r2 = st.columns(3)
        r2[0].text_input("ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", form["ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ"], key=f"{idx}_5")
        r2[1].text_input("ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", form["ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ"], key=f"{idx}_6")
        r2[2].text_input("ΚΑΤΟΙΚΙΑ", form["ΚΑΤΟΙΚΙΑ"], key=f"{idx}_7")

        # Table rows
        st.markdown("#### 📋 Πίνακας")
        for i, row in enumerate(form["TABLE_ROWS"]):
            st.text_input(f"Γραμμή {i}", row, key=f"{idx}_table_{i}")
