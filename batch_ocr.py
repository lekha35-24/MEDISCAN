import easyocr
import cv2
import re
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

BATCH_KEYWORDS = ["BNO", "BATCHNO", "BATCH", "LOTNO", "LOT", "LNO"]
EXCLUDE_KEYWORDS = ["MRP", "EXP", "MFG", "CAPS", "TABS", "TAX", "INCL",
                     "SCAN", "VERIFY", "QRCODE", "RS", "PRICE", "MANUFACTURED",
                     "STORAGE", "STORE", "PROTECT", "KEEP", "REACH", "CHILDREN"]

MIN_BATCH_LEN = 4
MAX_BATCH_LEN = 14


def preprocess_for_ocr(img):
    """
    Preprocessing pipeline based on documented best practices for
    packaging/label OCR: grayscale -> denoise -> CLAHE contrast
    enhancement -> adaptive threshold. This stabilizes character
    extraction against glare, low contrast, and uneven lighting
    common in phone-camera photos of foil/blister packaging.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise (median filter handles salt-and-pepper noise from foil glare well)
    denoised = cv2.medianBlur(gray, 3)

    # CLAHE: local contrast enhancement, much better than global
    # equalizeHist for uneven lighting across a curved blister pack
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Upscale — small printed batch/expiry text benefits from more pixels
    enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return enhanced


def get_ocr_blocks(image_path):
    """Run OCR once (on preprocessed image). Reuse for density + batch + expiry."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    processed = preprocess_for_ocr(img)
    # detail=1 gives bounding boxes too, which we use to keep batch-related
    # text fragments that are spatially close together
    return reader.readtext(processed, detail=1)


def get_text_density_from_blocks(blocks):
    return sum(len(b[1].strip()) for b in blocks)


def _is_plausible_batch(cleaned):
    if not (MIN_BATCH_LEN <= len(cleaned) <= MAX_BATCH_LEN):
        return False
    if not any(c.isdigit() for c in cleaned):
        return False
    if any(ex in cleaned for ex in EXCLUDE_KEYWORDS):
        return False
    # Reject strings that are ALL digits and very long (likely a phone
    # number, price, or merged multi-line OCR artifact)
    if cleaned.isdigit() and len(cleaned) > 10:
        return False
    return True


def extract_batch_number_from_blocks(blocks):
    if not blocks:
        return None

    # blocks: list of (bbox, text, confidence) from detail=1
    raw_texts = [b[1] for b in blocks]
    confidences = [b[2] for b in blocks]
    cleaned_texts = [re.sub(r'[^A-Z0-9]', '', t.strip().upper()) for t in raw_texts]

    # 1. Keyword-labeled batch number — highest priority, most reliable
    for i, cleaned in enumerate(cleaned_texts):
        for kw in BATCH_KEYWORDS:
            if cleaned.startswith(kw):
                remainder = cleaned[len(kw):]
                if _is_plausible_batch(remainder):
                    return remainder
                # Check next 1-2 OCR blocks — batch code is often split
                # across adjacent detected regions (e.g. "B.No." then "D12Z030")
                for j in range(i + 1, min(i + 3, len(cleaned_texts))):
                    if _is_plausible_batch(cleaned_texts[j]):
                        return cleaned_texts[j]

    # 2. Also check for keyword INSIDE a block, not just at the start
    #    (handles cases like "B.NO.AST-2506" glued together with no space)
    for i, cleaned in enumerate(cleaned_texts):
        for kw in BATCH_KEYWORDS:
            idx = cleaned.find(kw)
            if idx != -1:
                remainder = cleaned[idx + len(kw):]
                if _is_plausible_batch(remainder):
                    return remainder

    # 3. Fallback: plausible candidates, ranked by OCR confidence
    #    (prefer what EasyOCR itself is most confident about, not just length)
    candidates = [
        (cleaned_texts[i], confidences[i])
        for i in range(len(cleaned_texts))
        if _is_plausible_batch(cleaned_texts[i])
    ]
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None


# ---- Backward-compatible wrappers (main.py / real_world_eval.py unchanged) ----
def extract_batch_number(image_path):
    return extract_batch_number_from_blocks(get_ocr_blocks(image_path))


def get_text_density(image_path):
    return get_text_density_from_blocks(get_ocr_blocks(image_path))