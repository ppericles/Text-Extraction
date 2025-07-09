import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
from streamlit_drawable_canvas import st_canvas

# --- GCP Vision client ---
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = vision.ImageAnnotatorClient(credentials=credentials)

# 🔍 Debug mode toggle
debug_mode = st.sidebar.checkbox("🪪 Enable OCR Debug Overlay", value=False)

# --- Field extraction using absolute layout zones ---
def extract_forms_from_ocr(response, raw_blocks, debug=False):
    forms_data = []
    try:
        page = response.full_text_annotation.pages[0]
        img_width = page.width or 1000
        img_height = page.height or 1500
        form_height = img_height / 3

        # Optional debug: show recognized blocks with coordinates
        if debug:
            st.markdown("### 🧱 OCR Text Blocks with Coordinates")
            for b in raw_blocks:
                st.markdown(f"`{b['text']}` → (x={int(b['x'])}, y={int(b['y'])})")

            canvas_result = st_canvas(
                background_image=img,
                fill_color="rgba(255, 255, 0, 0.3)",
                stroke_width=1,
                update_streamlit=False,
                height=img.height,
                width=img.width,
                drawing_mode="transform",
                key="canvas_overlay",
            )

        def nearest_text(blocks, cx, cy, radius=100):
            found = [
                b for b in blocks
                if abs(b["x"] - cx) < radius and abs(b["y"] - cy) < radius
            ]
            if found:
                return sorted(found, key=lambda b: (abs(b["x"] - cx) + abs(b["y"] - cy)))[0]["text"]
            return ""

        for i in range(3):
            blocks = [b for b in raw_blocks if i * form_height <= b["y"] < (i + 1) * form_height]
            base_y = i * form_height

            fields = {
                "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": nearest_text(blocks, 100, base_y + 90),
                "ΕΠΩΝΥΜΟ": nearest_text(blocks, 260, base_y + 90),
                "ΚΥΡΙΟΝ ΟΝΟΜΑ": nearest_text(blocks, 470, base_y + 90),
                "ΟΝΟΜΑ ΠΑΤΡΟΣ": nearest_text(blocks, 650, base_y + 90),
                "ΟΝΟΜΑ ΜΗΤΡΟΣ": nearest_text(blocks, 820, base_y + 90),
                "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ": nearest_text(blocks, 200, base_y + 180),
                "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ": nearest_text(blocks, 470, base_y + 180),
                "ΚΑΤΟΙΚΙΑ": nearest_text(blocks, 740, base_y + 180),
                "TABLE_ROWS": [
                    b["text"] for b in blocks if b["y"] > base_y + 230 and len(b["text"].split()) >= 2
                ][:11]
            }
            forms_data.append(fields)

    except Exception as e:
        forms_data = [{"error": f"❌ OCR parsing failed: {e}"}]

    return forms_data

# --- App UI ---
st.set_page_config(layout="wide", page_title="Greek Form OCR")
st.title("📄 Greek OCR with Field Calibration Overlay")

uploaded_file = st.file_uploader("📎 Upload Greek form image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    if max(img.size) > 1800:
        img.thumbnail((1800, 1800))

    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Running Google OCR..."):
        uploaded_file.seek(0)
        image_proto = vision.Image(content=uploaded_file.read())
        try:
            response = client.document_text_detection(image=image_proto)
        except Exception as e:
            st.error(f"❌ Vision API error: {e}")
            st.stop()

        # Build flat block list with positions
        raw_blocks = []
        try:
            for block in response.full_text_annotation.pages[0].blocks:
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
        except Exception as e:
            st.error(f"Failed to read block data: {e}")
            st.stop()

        forms = extract_forms_from_ocr(response, raw_blocks, debug=debug_mode)

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
