import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
from streamlit_drawable_canvas import st_canvas
import json

# --- Google Cloud Vision credentials ---
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = vision.ImageAnnotatorClient(credentials=credentials)

# --- Default field layout ---
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

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Greek OCR Calibrator")
st.title("📄 Greek Form Parser with Live Field Layout Tuning")

# --- Layout Reimport UI ---
uploaded_layout = st.file_uploader("📂 Import Layout from JSON", type=["json"])
if uploaded_layout:
    try:
        loaded_layout = json.load(uploaded_layout)
        st.session_state.form_layouts = loaded_layout
        st.success("✅ Layout loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load layout: {e}")

# --- Sidebar UI for Calibration ---
st.sidebar.markdown("## 🛠️ Field Calibration")
form_number = st.sidebar.selectbox("📄 Select Form", [1, 2, 3])
field_label = st.sidebar.selectbox("📝 Field Name", list(default_positions.keys()))

if "form_layouts" not in st.session_state:
    st.session_state.form_layouts = {i: default_positions.copy() for i in [1, 2, 3]}

x_val, y_val = st.session_state.form_layouts[form_number][field_label]
x_val = st.sidebar.slider("X Position", 0, 1200, value=x_val)
y_val = st.sidebar.slider("Y Offset", 0, 400, value=y_val)
st.session_state.form_layouts[form_number][field_label] = (x_val, y_val)

# --- Image Upload and OCR ---
uploaded_file = st.file_uploader("📎 Upload scanned Greek form", type=["jpg", "jpeg", "png"])
if uploaded_file:
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)

    # ✅ Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # ✅ Resize safely
    img_width, img_height = img.size
    if img_width > 1800 or img_height > 1800:
        img = img.resize((min(img_width, 1800), min(img_height, 1800)))
        img_width, img_height = img.size

    st.image(img, caption="📷 Uploaded Form", use_column_width=True)

    with st.spinner("🔍 Performing OCR..."):
        uploaded_file.seek(0)
        image_proto = vision.Image(content=uploaded_file.read())

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

        form_height = img_height / 3
        forms = []
        overlays = []

        for i in [1, 2, 3]:
            base_y = (i - 1) * form_height
            layout = st.session_state.form_layouts[i]
            blocks_in_form = [b for b in blocks if base_y <= b["y"] < base_y + form_height]
            fields = {}

            for label, (rel_x, rel_y) in layout.items():
                cx, cy = rel_x, base_y + rel_y
                match = next(
                    (b for b in blocks_in_form if abs(b["x"] - cx) < 100 and abs(b["y"] - cy) < 100),
                    None
                )
                val = match["text"] if match else ""
                fields[label] = val

                if i == form_number:
                    overlays.append({
                        "label": f"{label}: {val or '(no match)'}",
                        "left": cx - 60,
                        "top": cy - 20,
                        "width": 120,
                        "height": 30
                    })

            rows = [b["text"] for b in blocks_in_form if b["y"] > base_y + 230 and len(b["text"].split()) >= 2][:11]
            fields["TABLE_ROWS"] = rows
            forms.append(fields)

    # --- Canvas Overlay
    st.markdown(f"### 🧭 Calibration Overlay – Φόρμα {form_number}")
    st_canvas(
        background_image=img,
        initial_drawing=overlays,
        height=img_height,
        width=img_width,
        update_streamlit=False,
        drawing_mode="transform",
        key="canvas_overlay"
    )

    # --- Export Tuned Layout
    st.download_button(
        label="💾 Download Layout as JSON",
        data=json.dumps(st.session_state.form_layouts, ensure_ascii=False, indent=2),
        file_name="form_layouts.json",
        mime="application/json"
    )

    # --- Display OCR Form Results
    for idx, form in enumerate(forms, start=1):
        with st.expander(f"📄 Φόρμα {idx}", expanded=(idx == form_number)):
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
