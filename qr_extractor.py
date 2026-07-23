import cv2
import re

# Try importing DataMatrix decoder
try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode
    DATAMATRIX_AVAILABLE = True
except:
    DATAMATRIX_AVAILABLE = False


# QR detectors
qr_detector_classic = cv2.QRCodeDetector()

try:
    _wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
except:
    _wechat_detector = None


def _preprocess_variants(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return [img, upscaled, clahe_img, thresh]


def _parse_gs1(text):
    batch = None
    expiry = None

    # Normalize GS1 (remove separators)
    text = text.replace("\x1d", "").replace(" ", "")

    # Batch (AI 10)
    batch_match = re.search(r"\(10\)([A-Z0-9]+)", text)
    if batch_match:
        batch = batch_match.group(1)

    # Expiry (AI 17 → YYMMDD)
    expiry_match = re.search(r"\(17\)(\d{6})", text)
    if expiry_match:
        raw = expiry_match.group(1)
        expiry = f"{raw[2:4]}/{raw[0:2]}"  # MM/YY

    return batch, expiry


def get_batch_and_expiry_from_qr(img):
    variants = _preprocess_variants(img)

    for variant in variants:

        # 1️⃣ WeChat QR
        if _wechat_detector is not None:
            try:
                results, _ = _wechat_detector.detectAndDecode(variant)
                payloads = [r for r in results if r]

                for text in payloads:
                    batch, expiry = _parse_gs1(text)
                    if batch or expiry:
                        return batch, expiry
            except:
                pass

        # 2️⃣ OpenCV QR multi
        try:
            retval, decoded_info, _, _ = qr_detector_classic.detectAndDecodeMulti(variant)
            if retval:
                for text in decoded_info:
                    if text:
                        batch, expiry = _parse_gs1(text)
                        if batch or expiry:
                            return batch, expiry
        except:
            pass

        # 3️⃣ OpenCV QR single  ✅ FIXED (your code was broken here)
        try:
            data, _, _ = qr_detector_classic.detectAndDecode(variant)
            if data:
                batch, expiry = _parse_gs1(data)
                if batch or expiry:
                    return batch, expiry
        except:
            pass

        # 4️⃣ DataMatrix (if available)
        if DATAMATRIX_AVAILABLE:
            try:
                results = dmtx_decode(variant)
                for res in results:
                    text = res.data.decode("utf-8")
                    batch, expiry = _parse_gs1(text)
                    if batch or expiry:
                        return batch, expiry
            except:
                pass

    return None, None