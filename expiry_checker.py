import cv2
import re
import numpy as np
import easyocr
from functools import lru_cache
from datetime import datetime

reader = easyocr.Reader(['en'], gpu=False)

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

EXPIRY_LABEL_PATTERN = re.compile(r'(EXP|EXPIRY|BEST\s*BEFORE|USE\s*BY|VALID\s*UNTIL)')
MFG_LABEL_PATTERN = re.compile(r'(MFG|MFD|PKD|PACKED)')

_DATE_TOKEN_PATTERN = re.compile(
    r'(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})|'
    r'(\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2})|'
    r'(\d{1,2}[/.\-]\d{2,4})|'
    r'(\d{4}[/.\-]\d{1,2})'
)

_MONTH_NAME_PATTERN = re.compile(
    r'(\d{1,2}\s*[-/. ]?\s*)?(' + '|'.join(MONTHS.keys()) + r')\s*[-/. ]?\s*(\d{2,4})'
)


# ---------------- OCR WITH ROTATIONS ---------------- #

def preprocess_for_ocr(img):
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(denoised)

    enhanced = cv2.resize(enhanced, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return enhanced


def _preprocess_variant_b(img):
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _sort_blocks_reading_order(blocks):
    def _key(b):
        box = b[0]
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        return (round(sum(ys) / len(ys), -1), sum(xs) / len(xs))
    return sorted(blocks, key=_key)


def _rotate_gray(gray, angle):
    if angle == 0:
        return gray
    elif angle == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    else:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _ocr_one_rotation(gray, angle):
    rotated = _rotate_gray(gray, angle)

    processed = preprocess_for_ocr(rotated)
    blocks = reader.readtext(processed, detail=1)

    score = sum(len(b[1]) * b[2] for b in blocks)
    if score < 1.0:
        processed_b = _preprocess_variant_b(rotated)
        blocks_b = reader.readtext(processed_b, detail=1)
        score_b = sum(len(b[1]) * b[2] for b in blocks_b)
        if score_b > score:
            blocks, score = blocks_b, score_b

    blocks = _sort_blocks_reading_order(blocks)
    return score, blocks


def _ocr_all_rotations(img):
    # Convert to grayscale ONCE instead of once per rotation — rotating a
    # single-channel array then using it gives identical pixel values to
    # rotating the BGR image and converting after, for 90/180/270 turns.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angles = [0, 90, 180, 270]

    results = [_ocr_one_rotation(gray, a) for a in angles]
    results.sort(key=lambda x: x[0], reverse=True)

    all_blocks = []
    for _, b in results:
        all_blocks.extend(b)

    return all_blocks


@lru_cache(maxsize=32)
def _get_ocr_blocks_cached(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return tuple()
    return tuple(_ocr_all_rotations(img))


def get_ocr_blocks(image_path):
    # Cached by image_path: if this exact path gets OCR'd twice in one
    # run (e.g. once for batch number, once for expiry), the second call
    # is free instead of re-running easyocr on all 4 rotations again.
    return list(_get_ocr_blocks_cached(image_path))


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


def _extract_date_from_text(text):
    for m in _DATE_TOKEN_PATTERN.finditer(text):
        token = m.group(0)
        parsed = _parse_date(token)
        if parsed:
            return parsed

    m = _MONTH_NAME_PATTERN.search(text)
    if m:
        month_str = m.group(2)
        year_str = m.group(3)
        month_num = MONTHS.get(month_str)
        if month_num:
            try:
                year = int(year_str)
                if year < 100:
                    year += 2000
                return datetime(year, month_num, 1)
            except ValueError:
                pass

    return None


# ---------------- SMART EXPIRY DETECTION ---------------- #

def extract_expiry_date(image_path):
    blocks = get_ocr_blocks(image_path)
    texts = [_normalize(b[1]) for b in blocks]

    candidates = []

    for i, text in enumerate(texts):
        if re.search(MFG_LABEL_PATTERN, text):
            continue

        if re.search(EXPIRY_LABEL_PATTERN, text):
            window = range(max(0, i - 2), min(i + 3, len(texts)))
            for j in window:
                parsed = _extract_date_from_text(texts[j])
                if parsed:
                    candidates.append(parsed)

    if not candidates:
        for text in texts:
            parsed = _extract_date_from_text(text)
            if parsed:
                candidates.append(parsed)

    if not candidates:
        return None

    return max(candidates).strftime("%m/%Y")


def check_expiry_validity(image_path):
    blocks = get_ocr_blocks(image_path)
    texts = [_normalize(b[1]) for b in blocks]

    expiry_date = None
    confidence = 0.0

    for i, text in enumerate(texts):
        if re.search(MFG_LABEL_PATTERN, text):
            continue

        if re.search(EXPIRY_LABEL_PATTERN, text):
            window = range(max(0, i - 2), min(i + 3, len(texts)))
            for j in window:
                parsed = _extract_date_from_text(texts[j])
                if parsed:
                    expiry_date = parsed
                    confidence = blocks[j][2]
                    break

        if expiry_date:
            break

    if not expiry_date:
        for i, text in enumerate(texts):
            parsed = _extract_date_from_text(text)
            if parsed:
                expiry_date = parsed
                confidence = blocks[i][2]
                break

    expiry_date_str = expiry_date.strftime("%m/%Y") if expiry_date else None

    if not expiry_date:
        return "NOT FOUND", 0.0, expiry_date_str

    today = datetime.today()

    if expiry_date < today:
        return "EXPIRED", confidence, expiry_date_str
    else:
        return "VALID", confidence, expiry_date_str
'''import cv2
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

# Matches numeric date-like tokens embedded anywhere in a string,
# e.g. "EXP05/26", "05-2026EXP", "12/05/2026"
_DATE_TOKEN_PATTERN = re.compile(
    r'(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})|'   # dd/mm/yyyy or dd/mm/yy
    r'(\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2})|'     # yyyy/mm/dd
    r'(\d{1,2}[/.\-]\d{2,4})|'                # mm/yyyy or mm/yy
    r'(\d{4}[/.\-]\d{1,2})'                   # yyyy/mm
)

# Matches "JAN2026", "JAN 26", "JAN-2026", "05 JAN 2026" style month-name dates
_MONTH_NAME_PATTERN = re.compile(
    r'(\d{1,2}\s*[-/. ]?\s*)?(' + '|'.join(MONTHS.keys()) + r')\s*[-/. ]?\s*(\d{2,4})'
)


# ---------------- OCR WITH ROTATIONS ---------------- #

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(denoised)

    # Stronger upscale — tiny embossed/printed strip text needs more than 2x
    # to clear OCR's reliable character-height threshold.
    enhanced = cv2.resize(enhanced, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return enhanced


def _preprocess_variant_b(img):
    """Fallback preprocessing pass — Otsu threshold instead of CLAHE.
    Some strip backgrounds (foil glare, low contrast print) respond
    better to a hard threshold than to contrast enhancement."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _sort_blocks_reading_order(blocks):
    """Sort OCR blocks top-to-bottom, left-to-right using bounding box
    centers, so label/date proximity search works on spatially adjacent
    text instead of arbitrary detection order."""
    def _key(b):
        box = b[0]
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        return (round(sum(ys) / len(ys), -1), sum(xs) / len(xs))  # row bucket, then x
    return sorted(blocks, key=_key)


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

        # Fallback preprocessing pass if the primary pass finds little/nothing
        score = sum(len(b[1]) * b[2] for b in blocks)
        if score < 1.0:
            processed_b = _preprocess_variant_b(rotated)
            blocks_b = reader.readtext(processed_b, detail=1)
            score_b = sum(len(b[1]) * b[2] for b in blocks_b)
            if score_b > score:
                blocks, score = blocks_b, score_b

        blocks = _sort_blocks_reading_order(blocks)
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


def _extract_date_from_text(text):
    """Pull a date out of a noisy OCR string instead of requiring the
    whole string to be a clean date. Handles both numeric dates
    (possibly glued to a label, e.g. 'EXP05/26') and month-name dates
    (e.g. 'EXP JUL 2026')."""
    # 1. Try numeric date substrings anywhere in the text
    for m in _DATE_TOKEN_PATTERN.finditer(text):
        token = m.group(0)
        parsed = _parse_date(token)
        if parsed:
            return parsed

    # 2. Try month-name dates using MONTHS
    m = _MONTH_NAME_PATTERN.search(text)
    if m:
        month_str = m.group(2)
        year_str = m.group(3)
        month_num = MONTHS.get(month_str)
        if month_num:
            try:
                year = int(year_str)
                if year < 100:
                    year += 2000
                return datetime(year, month_num, 1)
            except ValueError:
                pass

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

        # Look for expiry label — check the label's own block (handles
        # glued "EXP05/26"), then a small window forward AND backward,
        # since printed order varies ("EXP 05/26" vs "05/26 EXP").
        if re.search(EXPIRY_LABEL_PATTERN, text):
            window = range(max(0, i - 2), min(i + 3, len(texts)))
            for j in window:
                parsed = _extract_date_from_text(texts[j])
                if parsed:
                    candidates.append(parsed)

    # fallback (no label)
    if not candidates:
        for text in texts:
            parsed = _extract_date_from_text(text)
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

        # Look for expiry keyword — search both directions around the label
        if re.search(EXPIRY_LABEL_PATTERN, text):
            window = range(max(0, i - 2), min(i + 3, len(texts)))
            for j in window:
                parsed = _extract_date_from_text(texts[j])
                if parsed:
                    expiry_date = parsed
                    confidence = blocks[j][2]
                    break

        if expiry_date:
            break

    # Fallback: detect any date if no label found
    if not expiry_date:
        for i, text in enumerate(texts):
            parsed = _extract_date_from_text(text)
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
        return "VALID", confidence'''