import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
from streamlit_drawable_canvas import st_canvas

# --- Vision API client ---
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = vision.ImageAnnotatorClient(credentials=credentials)

# --- Default field slots (base layout for form 1)
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

# --- UI: Field Calibration Sidebar ---
st.sidebar.markdown("## 🛠️ Field Calibration")

form_number = st.sidebar.selectbox("📄 Select Form", [1, 2, 3])
field_label = st.sidebar.selectbox("📝 Field", list(default_positions.keys()))

if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: default_positions.copy() for i in [1, 2, 3]}

curr_x, curr_y = st.session_state.form_layouts[form_number][field_label]
x_val = st.sidebar.slider("X", 0, 1200, value=curr_x)
y_val = st.sidebar.slider("Y Offset (in Form)", 0, 400, value=curr_y)

st.session_state.form_layouts[form_number][field_label] = (x_val, y_val)

# --- Main App UI ---
st.set_page_config(layout="wide", page_title="Greek OCR Form Parser")
st.title("📄 Greek Form Parser with Field Overlay Calibration")

uploaded_file = st.file_uploader("📎 Upload Form Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    if max(img.size) > 1800:
        img.thumbnail((1800, 1800))
    img_width, img_height = img.size
    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 OCR in progress..."):
        uploaded_file.seek(0)
        image_proto = vision.Image(content=uploaded_file.read())
        try:
            response = client.document_text_detection(image=image_proto)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.stop()

        # Build text block list
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

        # --- Parser & Overlay Builder ---
        def nearest_text(blocks, cx, cy, radius=100):
            found = [b for b in blocks if abs(b["x"] - cx) < radius and abs(b["y"] - cy) < radius]
            if found:
                return sorted(found, key=lambda b: (abs(b["x"] - cx) + abs(b["y"] - cy)))[0]
            return None

        forms = []
        overlays = []
        form_height = img_height / 3

        for form_idx in [1, 2, 3]:
            base_y = (form_idx - 1) * form_height
            blocks_in_form = [b for b in blocks if base_y <= b["y"] < base_y + form_height]
            layout = st.session_state.form_layouts.get(form_idx, default_positions)
            fields = {}

            for label, (rel_x, rel_y) in layout.items():
                cx = rel_x
                cy = base_y + rel_y
                match = nearest_text(blocks_in_form, cx, cy)
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

            rows = [b["text"] for b in blocks_in_form if b["y"] > base_y + 230 and len(b["text"].split()) >= 2][:11]
            fields["TABLE_ROWS"] = rows
            forms.append(fields)

    # --- Overlay Canvas Preview ---
    st.markdown("### 🧭 Live Field Overlay")
    st_canvas(
        background_image=img,
        initial_drawing=overlays,
        height=img_height,
        width=img_width,
        update_streamlit=False,
        drawing_mode="transform",
        key="canvas_overlay"
    )

    # --- Display Extracted Forms ---
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
