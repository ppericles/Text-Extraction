# [same imports as before]
# ... (imports and setup unchanged)

st.set_page_config(layout="wide", page_title="Greek Registry Grid Extractor")
st.title("📜 Greek Registry Key-Value Grid Parser")

form_ids = [1, 2, 3]
labels_matrix = [
    ["ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ"],
    ["ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"]
]

# === Session
if "extracted_values" not in st.session_state:
    st.session_state.extracted_values = {}

# === Sidebar
cred_file = st.sidebar.file_uploader("🔐 Google credentials", type=["json"])
uploaded_file = st.file_uploader("📎 Upload registry scan", type=["jpg", "jpeg", "png"])

if cred_file:
    with open("credentials.json", "wb") as f:
        f.write(cred_file.read())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    st.sidebar.success("✅ Credentials loaded")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📄 Uploaded Registry", use_column_width=True)

# === Main processing
if uploaded_file and cred_file and st.button("🔍 Parse and Preview Forms"):
    np_image = np.array(image)
    height, width = np_image.shape[:2]
    form_height = height // 3
    left_width = width // 2
    pad = 5
    client = vision.ImageAnnotatorClient()

    for form_id in form_ids:
        y1, y2 = (form_id - 1) * form_height, form_id * form_height
        form_crop = np_image[y1:y2, :left_width].copy()
        gray = cv2.cvtColor(form_crop, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_not(binary)

        # Line detection
        scale = 20
        horiz = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1))), cv2.getStructuringElement(cv2.MORPH_RECT, (scale,1)))
        vert  = cv2.dilate(cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (1,scale))), cv2.getStructuringElement(cv2.MORPH_RECT, (1,scale)))

        lines_y = cv2.reduce(horiz, 1, cv2.REDUCE_MAX).reshape(-1)
        y_coords = [i for i in range(len(lines_y)) if lines_y[i] < 255]
        y_clusters = [y_coords[i] for i in range(0, len(y_coords), max(1, len(y_coords)//3))][:3]

        lines_x = cv2.reduce(vert, 0, cv2.REDUCE_MAX).reshape(-1)
        x_coords = [i for i in range(len(lines_x)) if lines_x[i] < 255]
        x_clusters = [x_coords[i] for i in range(0, len(x_coords), max(1, len(x_coords)//5))][:5]

        preview = Image.fromarray(form_crop)
        draw = ImageDraw.Draw(preview)
        form_data = {}

        st.subheader(f"📄 Φόρμα {form_id}")
        for r in range(2):
            for c in range(4):
                field = labels_matrix[r][c]
                try:
                    y_start = max(0, y_clusters[r] - pad)
                    y_end   = min(form_crop.shape[0], y_clusters[r+1] + pad if r+1 < len(y_clusters) else form_crop.shape[0])
                    x_start = max(0, x_clusters[c] - pad)
                    x_end   = min(form_crop.shape[1], x_clusters[c+1] + pad if c+1 < len(x_clusters) else form_crop.shape[1])
                    cell_crop = form_crop[y_start:y_end, x_start:x_end]

                    pil_img = Image.fromarray(cell_crop)
                    buffer = BytesIO()
                    pil_img.save(buffer, format="JPEG")
                    vision_img = types.Image(content=buffer.getvalue())
                    response = client.document_text_detection(image=vision_img)
                    raw_text = response.full_text_annotation.text.strip()
                    value = " ".join(raw_text.split("\n")).strip()
                    form_data[field] = value

                    draw.rectangle([(x_start, y_start), (x_end, y_end)], outline="red", width=2)
                    draw.text((x_start + 5, y_start + 5), f"{field}", fill="blue")
                except Exception as e:
                    form_data[field] = "—"

        st.session_state.extracted_values[str(form_id)] = form_data
        st.image(preview, caption=f"🖼️ Bounding Box Preview — Φόρμα {form_id}", use_column_width=True)

        # Editable Panel
        st.markdown(f"### ✏️ Review Φόρμα {form_id}")
        for field in labels_matrix[0] + labels_matrix[1]:
            current_val = form_data.get(field, "")
            corrected = st.text_input(f"{field}", value=current_val, key=f"{form_id}_{field}")
            st.session_state.extracted_values[str(form_id)][field] = corrected

# === Export
if st.session_state.extracted_values:
    st.markdown("## 💾 Export Final Data")
    json_data = json.dumps(st.session_state.extracted_values, indent=2, ensure_ascii=False)
    st.download_button("💾 Download JSON", data=json_data, file_name="registry_data.json", mime="application/json")

    table = []
    for fid, fields in st.session_state.extracted_values.items():
        row = {"Φόρμα": fid}
        row.update(fields)
        table.append(row)
    df = pd.DataFrame(table)
    csv_data = df.to_csv(index=False)
    st.download_button("📤 Download CSV", data=csv_data, file_name="registry_data.csv", mime="text/csv")
