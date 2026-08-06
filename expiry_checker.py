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

EXPIRY_LABEL_PATTERN = r'(EXP|EXPIRY|BEST\s*BEFORE|USE\s*BY|VALID\s*UNTIL)'
MFG_LABEL_PATTERN = r'(MFG|MFD|PKD|PACKED)'


# ---------------- OCR WITH ROTATIONS ---------------- #

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(denoised)

    enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return enhanced


def _ocr_all_rotations(img):
    all_blocks = []
    scored = []

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

        score = sum(len(b[1]) * b[2] for b in blocks)
        scored.append((score, blocks))

    scored.sort(key=lambda x: x[0], reverse=True)

    for _, b in scored:
        all_blocks.extend(b)

    return all_blocks


def get_ocr_blocks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []
    return _ocr_all_rotations(img)


# ---------------- DATE PARSING ---------------- #

def _normalize(text):
    return re.sub(r'[^A-Z0-9/.-]', '', text.upper())


def _parse_date(text):
    text = re.sub(r'[.\-\s]+', '/', text)

    formats = [
        "%m/%Y", "%m/%y",
        "%d/%m/%Y", "%d/%m/%y",
        "%Y/%m/%d", "%Y/%m"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except:
            continue

    return None


# ---------------- SMART EXPIRY DETECTION ---------------- #

def extract_expiry_date(image_path):
    blocks = get_ocr_blocks(image_path)
    texts = [_normalize(b[1]) for b in blocks]

    candidates = []

    for i, text in enumerate(texts):

        # Skip manufacturing
        if re.search(MFG_LABEL_PATTERN, text):
            continue

        # Look for expiry label
        if re.search(EXPIRY_LABEL_PATTERN, text):

            for j in range(i, min(i+3, len(texts))):
                parsed = _parse_date(texts[j])
                if parsed:
                    candidates.append(parsed)

    # fallback (no label)
    if not candidates:
        for text in texts:
            parsed = _parse_date(text)
            if parsed:
                candidates.append(parsed)

    if not candidates:
        return None

    # return most future date
    return max(candidates).strftime("%m/%Y")
def check_expiry_validity(image_path):
    blocks = get_ocr_blocks(image_path)
    texts = [_normalize(b[1]) for b in blocks]

    expiry_date = None
    confidence = 0.0

    for i, text in enumerate(texts):

        # Ignore manufacturing labels
        if re.search(MFG_LABEL_PATTERN, text):
            continue

        # Look for expiry keyword
        if re.search(EXPIRY_LABEL_PATTERN, text):

            for j in range(i, min(i+3, len(texts))):
                parsed = _parse_date(texts[j])
                if parsed:
                    expiry_date = parsed
                    confidence = blocks[j][2]
                    break

        if expiry_date:
            break

    # Fallback: detect any date if no label found
    if not expiry_date:
        for i, text in enumerate(texts):
            parsed = _parse_date(text)
            if parsed:
                expiry_date = parsed
                confidence = blocks[i][2]
                break

    if not expiry_date:
        return "NOT FOUND", 0.0

    today = datetime.today()

    if expiry_date < today:
        return "EXPIRED", confidence
    else:
        return "VALID", confidence