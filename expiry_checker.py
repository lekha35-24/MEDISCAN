import cv2
import re
import numpy as np
import easyocr
from datetime import datetime

reader = easyocr.Reader(['en'], gpu=False)

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}


def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return enhanced


def _ocr_all_rotations(img):
    """
    Real-world packaging photos are rarely perfectly upright — batch/expiry
    text often runs vertically along a side edge (as in your own photos).
    Try OCR at 0/90/180/270 degrees and keep whichever rotation gives
    EasyOCR the most total detected text (a solid proxy for 'this is the
    orientation where the printed label text is actually readable').
    """
    best_blocks = []
    best_score = -1

    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = img
        elif angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        processed = preprocess_for_ocr(rotated)
        blocks = reader.readtext(processed, detail=1)
        score = sum(len(b[1]) * b[2] for b in blocks)  # text length weighted by confidence

        if score > best_score:
            best_score = score
            best_blocks = blocks

    return best_blocks


def _try_parse_numeric(date_str):
    date_str = date_str.replace('-', '/')
    for fmt in ["%m/%Y", "%m/%y", "%d/%m/%Y", "%d/%m/%y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _try_parse_month_name(month_str, year_str):
    month_str = month_str[:3].upper()
    if month_str not in MONTHS:
        return None
    month = MONTHS[month_str]
    year_str = year_str.strip()
    try:
        year = int(year_str)
        if year < 100:
            year += 2000
        return datetime(year, month, 1)
    except (ValueError, TypeError):
        return None


def check_expiry_validity(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "NOT FOUND", 0.3

    # Scan the WHOLE image, not just a bottom crop — real photos have
    # the expiry text anywhere depending on how the package was held.
    blocks = _ocr_all_rotations(img)
    if not blocks:
        return "NOT FOUND", 0.3

    raw_texts = [b[1].strip().upper() for b in blocks]
    joined = " | ".join(raw_texts)

    # 1. Numeric date directly after EXP (allows punctuation between them:
    #    "EXP.02/2028", "EXP: 02/2028", "EXP 02/2028")
    exp_match = re.search(r'EXP[.:\s]*(\d{1,2}[/\-]\d{2,4})', joined)
    if exp_match:
        expiry_date = _try_parse_numeric(exp_match.group(1))
        if expiry_date:
            return ("VALID", 1.0) if expiry_date > datetime.now() else ("EXPIRED", 0.0)

    # 2. Month-name format: "EXP JUN 27", "EXP JUN2027", "EXP. JUN 2027"
    exp_month_match = re.search(
        r'EXP[.:\s]*([A-Z]{3,9})[\s.]*(\d{2,4})', joined
    )
    if exp_month_match:
        expiry_date = _try_parse_month_name(exp_month_match.group(1), exp_month_match.group(2))
        if expiry_date:
            return ("VALID", 1.0) if expiry_date > datetime.now() else ("EXPIRED", 0.0)

    # 3. "EXP" token found, but the date is in a nearby (not same) OCR block
    #    -- common when EasyOCR splits "EXP" and "07/2027" into separate boxes
    for i, tok in enumerate(raw_texts):
        if "EXP" in tok:
            for next_tok in raw_texts[i + 1:i + 3]:
                numeric_match = re.search(r'(\d{1,2}[/\-]\d{2,4})', next_tok)
                if numeric_match:
                    expiry_date = _try_parse_numeric(numeric_match.group(1))
                    if expiry_date:
                        return ("VALID", 1.0) if expiry_date > datetime.now() else ("EXPIRED", 0.0)

                month_match = re.search(r'([A-Z]{3,9})[\s.]*(\d{2,4})', next_tok)
                if month_match:
                    expiry_date = _try_parse_month_name(month_match.group(1), month_match.group(2))
                    if expiry_date:
                        return ("VALID", 1.0) if expiry_date > datetime.now() else ("EXPIRED", 0.0)

    # 4. No reliable EXP-labeled date found. Do NOT guess using an
    #    unlabeled date (e.g. MFG) — that produced false EXPIRED results before.
    return "NOT FOUND", 0.3