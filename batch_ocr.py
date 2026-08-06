import easyocr
import cv2
import re
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

BATCH_KEYWORDS = ["BNO", "BATCHNO", "BATCH", "LOTNO", "LOT", "LNO"]
EXPIRY_KEYWORDS = ["EXP", "EXPIRY", "EXPIRE", "EXPDATE"]

EXCLUDE_KEYWORDS = ["MRP", "MFG", "CAPS", "TABS", "TAX", "INCL",
                    "SCAN", "VERIFY", "QRCODE", "RS", "PRICE",
                    "MG", "ML"]  # 🔥 IMPORTANT FIX

MIN_BATCH_LEN = 4
MAX_BATCH_LEN = 14


# -----------------------------
# 🔥 STRONG PREPROCESSING
# -----------------------------
def preprocess_for_ocr(img):
    variants = []

    scales = [1.5, 2, 3]

    for scale in scales:
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Strong denoise
        denoise = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

        # Sharpen
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp = cv2.filter2D(denoise, -1, kernel)

        # CLAHE
        clahe = cv2.createCLAHE(2.5, (8, 8))
        enhanced = clahe.apply(sharp)

        # Threshold
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Morph close
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        variants.extend([enhanced, thresh, morph])

    return variants


# -----------------------------
# 🔥 OCR MULTI PASS
# -----------------------------
def get_ocr_blocks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []

    variants = preprocess_for_ocr(img)

    all_blocks = []
    for var in variants:
        blocks = reader.readtext(var, detail=1)
        for b in blocks:
            if b[2] > 0.35:
                all_blocks.append(b)

    return all_blocks


def get_text_density_from_blocks(blocks):
    return sum(len(b[1].strip()) for b in blocks)


# -----------------------------
# 🔥 CLEANING
# -----------------------------
def clean_text(text):
    return re.sub(r'[^A-Z0-9/.-]', '', text.upper())


def _is_plausible_batch(text):
    if not (MIN_BATCH_LEN <= len(text) <= MAX_BATCH_LEN):
        return False
    if not any(c.isdigit() for c in text):
        return False
    if any(ex in text for ex in EXCLUDE_KEYWORDS):
        return False
    return True


# -----------------------------
# 🔥 EXPIRY DATE EXTRACTION
# -----------------------------
def extract_expiry_from_blocks(blocks):
    if not blocks:
        return None

    texts = [clean_text(b[1]) for b in blocks]

    # 🔥 Common date patterns
    date_patterns = [
        r'(0[1-9]|1[0-2])[\/\-\.](20\d{2})',   # MM/YYYY
        r'(20\d{2})[\/\-\.](0[1-9]|1[0-2])',   # YYYY/MM
        r'(0[1-9]|[12][0-9]|3[01])[\/\-\.](0[1-9]|1[0-2])[\/\-\.](20\d{2})'
    ]

    # 1. Keyword based
    for i, text in enumerate(texts):
        for kw in EXPIRY_KEYWORDS:
            if kw in text:
                for j in range(i, min(i+3, len(texts))):
                    for pattern in date_patterns:
                        match = re.search(pattern, texts[j])
                        if match:
                            return match.group()

    # 2. Global scan
    for text in texts:
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()

    return None


# -----------------------------
# 🔥 BATCH EXTRACTION (FIXED)
# -----------------------------
def extract_batch_number_from_blocks(blocks):
    if not blocks:
        return None

    raw_texts = [b[1] for b in blocks]
    confidences = [b[2] for b in blocks]

    cleaned_texts = [clean_text(t) for t in raw_texts]

    # 1. Keyword-based
    for i, text in enumerate(cleaned_texts):
        for kw in BATCH_KEYWORDS:
            if kw in text:
                idx = text.find(kw)
                remainder = text[idx + len(kw):]

                if _is_plausible_batch(remainder):
                    return remainder

                # Check neighbors
                for j in range(i+1, min(i+4, len(cleaned_texts))):
                    if _is_plausible_batch(cleaned_texts[j]):
                        return cleaned_texts[j]

    # 2. Confidence-ranked fallback
    candidates = [
        (cleaned_texts[i], confidences[i])
        for i in range(len(cleaned_texts))
        if _is_plausible_batch(cleaned_texts[i])
    ]

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None


# -----------------------------
# ✅ SAME WRAPPERS
# -----------------------------
def extract_batch_number(image_path):
    return extract_batch_number_from_blocks(get_ocr_blocks(image_path))


def extract_expiry_date(image_path):   # 🔥 NEW FUNCTION
    return extract_expiry_from_blocks(get_ocr_blocks(image_path))


def get_text_density(image_path):
    return get_text_density_from_blocks(get_ocr_blocks(image_path))