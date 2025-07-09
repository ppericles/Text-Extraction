import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import io

# ---------------------------
# Google Vision Client
# ---------------------------

credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = vision.ImageAnnotatorClient(credentials=credentials)

# ---------------------------
# Form Extraction from OCR
# ---------------------------

def extract_forms_from_ocr(response):
    forms_data = []

    try:
        page = response.full_text_annotation.pages[0]
        image_height = page.height or 1000
        form_height = image_height / 3

        raw_blocks = []
        for block in page.blocks:
            for para in block.paragraphs:
                text = ""
                for word in para.words:
                    word_text = "".join([s.text for s in word.symbols])
                    text += word_text + " "
                text = text.strip()
                y_center = sum(v.y for v in block.bounding_box.vertices) / 4
                x_center = sum(v.x for v in block.bounding_box.vertices) / 4
                raw_blocks.append({"text": text, "x": x_center, "y": y_center})

        forms = {1: [], 2: [], 3: []}
        for b in raw_blocks:
            form_idx = int(b["y"] // form_height) + 1
            if 1 <= form_idx <= 3:
                forms[form_idx].append(b)

        for idx in range(1, 4):
            form_blocks = forms[idx]
            fields = {
                "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": "", "ΕΠΩΝΥΜΟ": "", "ΚΥΡΙΟΝ ΟΝΟΜΑ": "",
                "ΟΝΟΜΑ ΠΑΤΡΟΣ": "", "ΟΝΟΜΑ ΜΗΤΡΟΣ": "",
                "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ": "", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ": "", "ΚΑΤΟΙΚΙΑ": "",
                "TABLE_ROWS": []
            }

            for block in form_blocks:
                txt = block["text"]

                if "ΜΕΡΙΔΟΣ" in txt or txt.isdigit() and len(txt) == 6:
                    fields["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"] = txt
                elif txt.isupper() and len(txt.split()) == 1:
                    if not fields["ΕΠΩΝΥΜΟ"]:
                        fields["ΕΠΩΝΥΜΟ"] = txt
                    elif not fields["ΚΥΡΙΟΝ ΟΝΟΜΑ"]:
                        fields["ΚΥΡΙΟΝ ΟΝΟΜΑ"] = txt
                elif "ΠΑΤΡΟΣ" in txt or "ΓΕΩΡΓΙΟΣ" in txt:
                    fields["ΟΝΟΜΑ ΠΑΤΡΟΣ"] = txt
                elif "ΜΗΤΡΟΣ" in txt or "ΕΛΕΝΗ" in txt:
                    fields["ΟΝΟΜΑ ΜΗΤΡΟΣ"] = txt
                elif "ΑΘΗΝΑ" in txt or "ΠΕΙΡΑΙΑΣ" in txt:
                    fields["ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ"] = txt
                elif txt.isdigit() and 1900 <= int(txt) <= 2025:
                    fields["ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ"] = txt
                elif "ΟΔΟΣ" in txt or "ΑΓ" in txt:
                    fields["ΚΑΤΟΙΚΙΑ"] = txt
                elif len(txt.split()) >= 2:
                    fields["TABLE_ROWS"].append(txt)

            fields["TABLE_ROWS"] = fields["TABLE_ROWS"][:11]
            forms_data.append(fields)

    except Exception as e:
        forms_data = [{"error": f"Error parsing forms: {e}"}]

    return forms_data

# ---------------------------
# Streamlit Interface
# ---------------------------

st.set_page_config(page_title="Greek Form OCR", layout="wide")
st.title("🇬🇷 Greek Handwriting OCR – Form Parser")

uploaded_file = st.file_uploader("📄 Upload handwritten form image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    st.markdown("---")

    col1, _ = st.columns([1.5, 2])
    with col1:
        st.image(img, caption="📸 Uploaded Form", use_column_width=True)

    with st.spinner("🧠 Processing OCR..."):
        content = uploaded_file.read()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        forms = extract_forms_from_ocr(response)

    for idx, form in enumerate(forms, start=1):
        st.markdown(f"## 📄 Φόρμα {idx}")
        if "error" in form:
            st.error(form["error"])
            continue

        # --- Line 1 ---
        line1 = st.columns(5)
        line1[0].text_input("ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", form["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ"], key=f"{idx}_1")
        line1[1].text_input("ΕΠΩΝΥΜΟ", form["ΕΠΩΝΥΜΟ"], key=f"{idx}_2")
        line1[2].text_input("ΚΥΡΙΟΝ ΟΝΟΜΑ", form["ΚΥΡΙΟΝ ΟΝΟΜΑ"], key=f"{idx}_3")
        line1[3].text_input("ΟΝΟΜΑ ΠΑΤΡΟΣ", form["ΟΝΟΜΑ ΠΑΤΡΟΣ"], key=f"{idx}_4")
        line1[4].text_input("ΟΝΟΜΑ ΜΗΤΡΟΣ", form["ΟΝΟΜΑ ΜΗΤΡΟΣ"], key=f"{idx}_5")

        # --- Line 2 ---
        line2 = st.columns(3)
        line2[0].text_input("ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", form["ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ"], key=f"{idx}_6")
        line2[1].text_input("ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", form["ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ"], key=f"{idx}_7")
        line2[2].text_input("ΚΑΤΟΙΚΙΑ", form["ΚΑΤΟΙΚΙΑ"], key=f"{idx}_8")

        st.markdown("#### 📊 Πίνακας Δεδομένων")
        for i, row in enumerate(form["TABLE_ROWS"]):
            st.text_input(f"🔹 Γραμμή {i}", row, key=f"{idx}_table_{i}")
