# ==== utils_text.py ====

import re, unicodedata
from datetime import datetime

# 🔡 Normalize Latin → Greek substitutions
def fix_latin_greek(text):
    latin_to_greek = {
        "A": "Α", "B": "Β", "E": "Ε", "H": "Η", "K": "Κ", "M": "Μ",
        "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ", "Y": "Υ"
    }
    return "".join(latin_to_greek.get(c, c) for c in text)

# 🔡 Normalize Cyrillic → Greek substitutions
def fix_cyrillic_greek(text):
    cyrillic_to_greek = {
        "А": "Α", "В": "Β", "С": "Σ", "Е": "Ε", "Н": "Η", "К": "Κ",
        "М": "Μ", "О": "Ο", "Р": "Ρ", "Т": "Τ", "Х": "Χ"
    }
    return "".join(cyrillic_to_greek.get(c, c) for c in text)

# 🧼 Full Text Normalization Pipeline
def normalize(text):
    if not text: return ""
    text = fix_latin_greek(text)
    text = fix_cyrillic_greek(text)
    text = unicodedata.normalize("NFD", text)
    text = ''.join(c for c in text if unicodedata.category(c) != "Mn")  # remove accents
    text = re.sub(r"[^\w\sΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]", "", text)  # remove punctuation
    return text.upper().strip()

# 📅 Normalize Greek Dates to DD/MM/YYYY
def normalize_date(text):
    text = text.strip()
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y"]:
        try: return datetime.strptime(text, fmt).strftime("%d/%m/%Y")
        except: continue
    return text  # fallback if no format matched

# 🛡️ Field Validation Logic
def validate_registry_field(label, corrected_text, confidence):
    issues = []
    greek_chars = re.findall(r"[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ]", corrected_text or "")
    if not corrected_text: issues.append("Missing")
    if label != "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ" and len(greek_chars) < max(3, len(corrected_text) // 2):
        issues.append("Non-Greek characters")
    if len(corrected_text) < 2: issues.append("Too short")
    if confidence < 50.0: issues.append("Low confidence")
    return issues

# 💡 Suggestion Generator
def suggest_fix(label, corrected_text, issues):
    if "Too short" in issues or "Non-Greek characters" in issues:
        fixed = corrected_text.title()
        if len(fixed) >= 2 and re.match(r"^[Α-ΩΆΈΉΊΌΎΏ][α-ωάέήίόύώ]{2,}", fixed): return fixed
    return None

# 📈 Heuristic Confidence Estimator
def estimate_confidence(label, text):
    text = text.strip()
    if not text: return 0.0
    if label == "ΑΡΙΘΜΟΣ ΜΕΡΙΔΟΣ":
        return 90.0 if text.isdigit() else 40.0
    if label in ["ΕΠΩΝΥΜΟΝ", "ΟΝΟΜΑ ΠΑΤΡΟΣ", "ΟΝΟΜΑ ΜΗΤΡΟΣ", "ΚΥΡΙΟΝ ΟΝΟΜΑ"]:
        return 75.0 if re.match(r"^[Α-ΩΆΈΉΊΌΎΏα-ωάέήίόύώ\s\-]{3,}$", text) else 30.0
    return 50.0
