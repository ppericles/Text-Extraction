import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
from streamlit_drawable_canvas import st_canvas

# --- Initialize Vision client ---
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = vision.ImageAnnotatorClient(credentials=credentials)

# --- Default coordinates for each header field (customizable) ---
default_positions = {
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ": (100, 90),
    "ΕΠΩΝΥΜΟ": (260, 90),
    "ΚΥΡΙΟΝ ΟΝΟΜΑ": (470, 90),
    "ΟΝΟΜΑ ΠΑΤΡΟΣ": (650, 90),
    "ΟΝΟΜΑ ΜΗΤΡΟΣ": (820, 90),
    "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ": (200, 180),
    "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ": (470, 180),
    "ΚΑΤΟΙΚΙΑ": (740, 180),
}

# --- Sidebar tuner ---
st.sidebar.markdown("## 🛠️ Field Coordinate Tuning")

field_to_tune = st.sidebar.selectbox("🔧 Choose Field", list(default_positions.keys()))

if "field_positions" not in st.session_state:
    st.session_state.field_positions = default_positions.copy()

curr_x, curr_y = st.session_state.field_positions[field_to_tune]

new_x = st.sidebar.slider("X", 0, 1200, value=curr_x)
new_y = st.sidebar.slider("Y", 0, 1500, value=curr_y)
st.session_state.field_positions[field_to_tune] = (new_x, new_y)

# --- OCR + parser ---
def extract_forms(response, raw_blocks, img_size):
    forms_data = []
    overlays = []
    img_width, img_height = img_size
    form_height = img_height / 3

    def nearest_text(blocks, cx, cy, radius=100):
        found = [
            b for b in blocks
            if abs(b["x"] - cx) < radius and abs(b["y"] - cy) < radius
        ]
        if found:
            return sorted(found, key=lambda b: (abs(b["x"] - cx) + abs(b["y"] - cy)))[0]
        return None

    for form_idx in range(3):
        blocks = [b for b in raw_blocks if form_idx * form_height <= b["y"] < (form_idx + 1) * form_height]
        base_y = form_idx * form_height

        fields = {}
        for label, (rel_x, rel_y) in st.session_state.field_positions.items():
            cx, cy = rel_x, base_y + rel_y
            match = nearest_text(blocks, cx, cy)
            val = match["text"] if match else ""
            fields[label] = val

            if match:
                overlays.append({
                    "label": f"{label}: {val}",
                    "left": match["x"] - 60,
                    "top": match["y"] - 20,
                    "width": 120,
                    "height": 30
                })

        fields["TABLE_ROWS"] = [
            b["text"] for b in blocks if b["y"] > base_y + 230 and len(b["text"].split()) >= 2
        ][:11]

        forms_data.append(fields)

    return forms_data, overlays

# --- App UI ---
st.set_page_config(layout="wide", page_title="Greek Form Calibrator")
st.title("📄 Greek OCR Form Parser with Live Field Calibration")

uploaded_file = st.file_uploader("📎 Upload scanned form", type=["png", "jpg", "jpeg"])
if uploaded_file:
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    if max(img.size) > 1800:
        img.thumbnail((1800, 1800))
    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 OCR in progress..."):
        uploaded_file.seek(0)
        content = uploaded_file.read()
        image_proto = vision.Image(content=content)

        try:
            response = client.document_text_detection(image=image_proto)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.stop()

        blocks = []
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
                blocks.append({"text": text, "x": x, "y": y})

        forms, overlays = extract_forms(response, blocks, img.size)

    # Show overlay
    st.markdown("### 🧭 Field Overlay Preview")
    st_canvas(
        background_image=img,
        initial_drawing=overlays,
        height=img.height,
        width=img.width,
        update_streamlit=False,
        drawing_mode="transform",
        key="overlay_canvas",
    )

    for idx, form in enumerate(forms, start=1):
        with st.expander(f"📄 Φόρμα {idx}", expanded=(idx == 1)):
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
