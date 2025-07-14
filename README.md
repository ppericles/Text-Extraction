
# 🇬🇷 Greek OCR Annotator & Template Builder

A modular Streamlit-based app suite for tagging scanned Greek documents, performing OCR, and building reusable field layouts. Supports `.jp2`, `.jpg`, `.png`, and batch processing across forms with smart field extraction.

---

## ⚡️ Features

### `app.py` — OCR Annotator
- 📤 Upload scanned forms with auto `.jp2` conversion
- 🧠 OCR via Google Vision API
- 👆 Click-based field tagging with smart label suggestions
- 🖼️ Preview extracted fields with bounding boxes
- 📊 Form comparison dashboard
- 💾 Export layout as JSON

### `template_creator.py` — Quick Template Generator
- 🖱️ Click-to-tag fields in a single form
- 🏷️ Assign labels from dropdowns
- 📤 Export layout JSON for use in main annotator

### `layout_template_builder.py` — Advanced Builder (Modular)
- 📁 Batch tagging across multiple forms
- 🖼️ Live preview canvas per form
- 📊 Completeness tracker with emoji indicators
- ✏️ Stubbed modules for:
  - Draggable/resizable boxes
  - Template auto-alignment
  - Smart label prediction

---

## 📂 Project Structure

```plaintext
ocr-annotator-project/
│
├── app.py
├── template_creator.py
├── layout_template_builder.py
│
├── utils/
│   ├── image_utils.py
│   ├── ocr_utils.py
│   ├── layout_utils.py
│
├── data/
│   ├── scanned_forms/
│   ├── form_layouts.json
│   └── credentials.json
├── assets/
│   └── example_form_1.jpg
│
├── README.md
└── requirements.txt
