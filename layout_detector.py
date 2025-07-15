# layout_detector.py

import layoutparser as lp
import cv2
from PIL import Image
from io import BytesIO
from google.cloud import vision

field_labels = [
    "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ", "ΕΠΩΝΥΜΟ", "ΚΥΡΙΟΝ ΟΝΟΜΑ", "ΟΝΟΜΑ ΠΑΤΡΟΣ",
    "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΤΟΠΟΣ ΓΕΝΝΗΣΕΩΣ", "ΕΤΟΣ ΓΕΝΝΗΣΕΩΣ", "ΚΑΤΟΙΚΙΑ"
]

def detect_layout_and_extract_fields(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = lp.models.Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6]
    )
    layout = model.detect(image_rgb)

    # Group blocks by Y position into vertical forms
    forms = {"1": [], "2": [], "3": []}
    for block in layout:
        y_mid = (block.coordinates[1] + block.coordinates[3]) // 2
        if y_mid < 900: forms["1"].append(block)
        elif y_mid < 1600: forms["2"].append(block)
        else: forms["3"].append(block)

    client = vision.ImageAnnotatorClient()
    form_layouts = {}

    def extract_text(block):
        x1, y1, x2, y2 = map(int, block.coordinates)
        crop = image_rgb[y1:y2, x1:x2]
        pil_img = Image.fromarray(crop)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        content = buf.getvalue()
        response = client.document_text_detection(image=vision.Image(content=content))
        return response.full_text_annotation.text.strip()

    for fid, blocks in forms.items():
        form_layouts[fid] = {}
        for block in blocks:
            text = extract_text(block)
            for label in field_labels:
                if label in text:
                    x1, y1, x2, y2 = map(int, block.coordinates)
                    form_layouts[fid][label] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    break

    return form_layouts
