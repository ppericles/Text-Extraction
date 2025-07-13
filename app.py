import streamlit as st
from PIL import Image, ImageDraw
import json
import os
import numpy as np
from google.cloud import vision
from streamlit_drawable_canvas import st_canvas
from io import BytesIO

# Mock utils if not available (remove these if you have the actual utils)
def normalize(text):
    return text.upper().strip()

def detect_header_regions(annotations, field_labels, field_boxes, debug=False):
    pass

def compute_form_bounds(field_boxes):
    if not field_boxes:
        return None
    x_coords = [box['x1'] for box in field_boxes.values()] + [box['x2'] for box in field_boxes.values()]
    y_coords = [box['y1'] for box in field_boxes.values()] + [box['y2'] for box in field_boxes.values()]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def image_to_base64(image):
    from io import BytesIO
    import base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Initialize session state properly
def init_session_state():
    if "form_layouts" not in st.session_state:
        st.session_state.form_layouts = {1: {}, 2: {}, 3: {}}
    if "ocr_blocks" not in st.session_state:
        st.session_state.ocr_blocks = []
    if "auto_extracted_fields" not in st.session_state:
        st.session_state.auto_extracted_fields = {}

init_session_state()

# App layout
st.set_page_config(layout="wide", page_title="Greek OCR Annotator")
st.title("🇬🇷 Greek OCR Annotator — Drag-to-Tag Edition")

field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

# Sidebar controls
form_num = st.sidebar.selectbox("📄 Φόρμα", [1, 2, 3])
selected_label = st.sidebar.selectbox("📝 Select Field Label", field_labels)

# Credentials upload
cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
if cred_file:
    try:
        with open("credentials.json", "wb") as f:
            f.write(cred_file.getvalue())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
        st.sidebar.success("✅ Credentials loaded")
    except Exception as e:
        st.sidebar.error(f"Credentials error: {str(e)}")

# Layout import
layout_file = st.sidebar.file_uploader("📂 Import layout (.json)", type=["json"])
if layout_file:
    try:
        st.session_state.form_layouts = json.load(layout_file)
        st.sidebar.success("✅ Layout imported")
    except Exception as e:
        st.sidebar.error(f"Import failed: {str(e)}")

# Main image uploader
uploaded_file = st.file_uploader("📎 Upload scanned form", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Read the image file
        image = Image.open(uploaded_file).convert("RGB")
        width, height = image.size
        
        # Store file bytes for OCR
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        # Get or create form layout
        if form_num not in st.session_state.form_layouts:
            st.session_state.form_layouts[form_num] = {}
        field_boxes = st.session_state.form_layouts[form_num]
        
        # Canvas setup
        st.markdown("### 🖱️ Drag to Tag Field Regions")
        
        # Create a fresh image for canvas background
        bg_image = Image.open(BytesIO(file_bytes)).convert("RGB")
        
        canvas_kwargs = {
            "fill_color": "rgba(0, 255, 0, 0.3)",
            "stroke_width": 2,
            "stroke_color": "green",
            "height": height,
            "width": width,
            "drawing_mode": "rect",
            "key": f"canvas_{form_num}",
            "background_image": bg_image,
            "update_streamlit": True
        }

        # Drawable canvas with error handling
        try:
            canvas_result = st_canvas(**canvas_kwargs)
            
            # Handle drawing results
            if canvas_result and canvas_result.json_data and canvas_result.json_data.get("objects"):
                latest = canvas_result.json_data["objects"][-1]
                x1 = int(latest["left"])
                y1 = int(latest["top"])
                x2 = x1 + int(latest["width"])
                y2 = y1 + int(latest["height"])
                
                field_boxes[selected_label] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                st.session_state.form_layouts[form_num] = field_boxes
                st.success(f"✅ Box saved for '{selected_label}' in Φόρμα {form_num}")
                
        except Exception as e:
            st.error(f"Canvas error: {str(e)}")
            st.stop()

        # OCR Processing
        if cred_file and os.path.exists("credentials.json"):
            try:
                with st.spinner("🔍 Running OCR..."):
                    # Initialize Google Vision client
                    client = vision.ImageAnnotatorClient()
                    vision_img = vision.Image(content=file_bytes)
                    
                    # Perform text detection
                    response = client.document_text_detection(image=vision_img)
                    annotations = response.text_annotations
                    
                    # Process annotations if available
                    if annotations:
                        detect_header_regions(annotations, field_labels, field_boxes, debug=True)
                        
                        # Create annotated image
                        draw_img = image.copy()
                        draw = ImageDraw.Draw(draw_img)
                        blocks = []
                        
                        # Process each annotation
                        for ann in annotations[1:]:
                            vertices = ann.bounding_poly.vertices
                            xs = [int(v.x) for v in vertices if v.x is not None]
                            ys = [int(v.y) for v in vertices if v.y is not None]
                            
                            if xs and ys:
                                x1, x2 = min(xs), max(xs)
                                y1, y2 = min(ys), max(ys)
                                center = (sum(xs) / len(xs), sum(ys) / len(ys))
                                blocks.append({"text": ann.description, "center": center})
                                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=1)
                                draw.text((x1, y1 - 10), ann.description, fill="blue")
                        
                        st.session_state.ocr_blocks = blocks
                        
                        # Draw field boxes
                        for label in field_labels:
                            box = field_boxes.get(label)
                            if box:
                                x1, y1 = box["x1"], box["y1"]
                                x2, y2 = box["x2"], box["y2"]
                                draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
                                draw.text((x1, y1 - 12), label, fill="green")
                        
                        # Display annotated image
                        st.markdown("### 📌 Tagged Overlay Image")
                        overlay_base64 = image_to_base64(draw_img)
                        st.markdown(
                            f"""
                            <div style='border:1px solid #ccc; overflow-x:auto; padding:10px'>
                                <img src='data:image/png;base64,{overlay_base64}' style='max-height:800px' />
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Show extracted field values
                        st.subheader("🧠 Extracted Field Values")
                        for i in [1, 2, 3]:
                            st.markdown(f"### 📄 Φόρμα {i}")
                            layout = st.session_state.form_layouts.get(i, {})
                            for label in field_labels:
                                box = layout.get(label)
                                if box:
                                    xmin, xmax = sorted([box["x1"], box["x2"]])
                                    ymin, ymax = sorted([box["y1"], box["y2"]])
                                    matches = [
                                        b["text"] for b in st.session_state.ocr_blocks
                                        if xmin <= b["center"][0] <= xmax and ymin <= b["center"][1] <= ymax
                                    ]
                                    val = " ".join(matches) if matches else "(no match)"
                                    st.text_input(label, val, key=f"{i}_{label}")

            except Exception as e:
                st.error(f"OCR processing failed: {str(e)}")
                st.exception(e)

        # Export layout
        st.download_button(
            label="💾 Export Layout as JSON",
            data=json.dumps(st.session_state.form_layouts, indent=2, ensure_ascii=False),
            file_name="form_layouts.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        st.exception(e)
