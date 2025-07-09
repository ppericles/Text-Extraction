import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account

# --- Google Vision API Client ---
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = vision.ImageAnnotatorClient(credentials=credentials)

# --- Absolute Field Slot Matching ---
def extract_forms_from_ocr(response):
    forms_data = []
    try:
        page = response.full_text_annotation.pages[0]
        img_width = page.width or 1000
        img_height = page.height or 1500
        form_height = img_height / 3

        raw_blocks = []
        for block in page.blocks:
            text = ""
            for para in block.paragraphs:
                for word in para.words:
                    text += ''.join(s.text for s in word.symbols) + " "
            text = text.strip()
            if text:
                x = sum(v.x for v in block.bounding_box.vertices) / 4
                y = sum(v.y for v in block.bounding_box.vertices) / 4
                raw_blocks.append({"text": text, "x": x, "y": y})

        # Define expected field positions by zone
        def nearest_text(form_blocks, center_x, center_y, radius=100):
            candidates = [
                b for b in form_blocks
                if abs(b["x"] - center_x) < radius and abs(b["y"] - center_y) < radius
            ]
            if candidates:
                return sorted(candidates, key=lambda b: (abs(b["x"] - center_x) + abs(b["y"] - center_y)))[0]["text"]
            return ""

        for i in range(3):
            blocks = [b for b in raw_blocks if i * form_height <= b["y"] < (i + 1) * form_height]
            base_y = i * form_height

            fields = {
                "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": nearest_text(blocks, 100, base_y + 90),
                "ΕΠΩΝΥΜΟ": nearest_text(blocks, 250, base_y + 90),
                "ΚΥΡΙΟΝ ΟΝΟΜΑ": nearest_text(blocks, 430, base_y + 90),
                "ΟΝΟΜΑ ΠΑΤΡΟΣ": nearest_text(blocks, 610, base_y + 90),
                "ΟΝΟΜΑ ΜΗΤΡΟΣ": nearest_text(blocks, 800, base_y + 90),
                "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ": nearest_text(blocks, 200, base_y + 180),
                "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ": nearest_text(blocks, 450, base_y + 180),
                "ΚΑΤΟΙΚΙΑ": nearest_text(blocks, 700, base_y + 180),
                "TABLE_ROWS": [
                    b["text"] for b in blocks if b["y"] > base_y + 230 and len(b["text"].split()) >= 2
                ][:11]
            }

            forms_data.append(fields)

    except Exception as e:
        forms_data = [{"error": f"❌ OCR parsing failed: {e}"}]

    return forms_data

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Greek Form Parser")
st.title("📄 Greek OCR – Absolute Layout Form Parser")

uploaded_file = st.file_uploader("📎 Upload Greek form image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    if max(img.size) > 1800:
        img.thumbnail((1800, 1800))

    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 OCR and extraction..."):
        uploaded_file.seek(0)
        image_proto = vision.Image(content=uploaded_file.read())
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
