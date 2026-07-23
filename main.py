import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime


from counterfeit_model import predict_counterfeit, load_model, grad_cam, transform, device
from batch_ocr import extract_batch_number, get_text_density
from expiry_checker import check_expiry_validity
from qr_extractor import get_batch_and_expiry_from_qr
#print("WeChat QR detector loaded:", _wechat_detector is not None)
model = load_model()

HIGH_CONF = 0.90
MED_CONF = 0.80
TEXT_DENSITY_THRESHOLD = 25


def resolve_expiry_status_from_qr_string(expiry_str):
    """Converts a QR-extracted 'MM/YYYY' string into the same
    ('VALID'/'EXPIRED'/'NOT FOUND', confidence) format expiry_checker.py uses,
    so the rest of the pipeline doesn't need to change."""
    if expiry_str is None:
        return "NOT FOUND", 0.3

    for fmt in ["%m/%Y", "%m/%y"]:
        try:
            expiry_date = datetime.strptime(expiry_str, fmt)
            return ("VALID", 1.0) if expiry_date > datetime.now() else ("EXPIRED", 0.0)
        except ValueError:
            continue

    return "NOT FOUND", 0.3


def decide(result, conf, batch, expiry_status, is_back_side):
    if is_back_side:
        if batch is None and expiry_status == "NOT FOUND":
            trust_note = " (OCR/QR unreadable — relying on visual model)"
        else:
            trust_note = ""

        if expiry_status == "EXPIRED":
            return "REJECTED: Expired"

        if result == "Fake" and conf > HIGH_CONF:
            return "HIGHLY SUSPICIOUS" + trust_note
        if result == "Fake" and conf > MED_CONF:
            return "SUSPICIOUS" + trust_note
        if result == "Real" and conf > HIGH_CONF:
            return "LIKELY GENUINE" + trust_note
        return "UNCERTAIN" + trust_note
    else:
        if result == "Fake" and conf > HIGH_CONF:
            return "SUSPICIOUS (visual only, no OCR/QR verification)"
        if result == "Real" and conf > HIGH_CONF:
            return "LIKELY GENUINE (visual only, no OCR/QR verification)"
        return "UNCERTAIN (visual only — consider photographing back of strip or QR code)"


def run_grad_cam(img, path_label=""):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        heatmap = grad_cam(model, img_tensor)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        img_resized = cv2.resize(img, (224, 224))
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)

        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Grad-CAM Heatmap")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print("Grad-CAM failed:", e)


if __name__ == "__main__":
    print("1. Check Medicine")
    choice = input("Enter 1: ")

    if choice == "1":
        path = input("Enter image path: ")
        if not os.path.exists(path):
            print("Image not found")
            exit()

        img = cv2.imread(path)

        text_density = get_text_density(path)
        is_back_side = text_density >= TEXT_DENSITY_THRESHOLD
        print(f"\nText density: {text_density} ({'back side' if is_back_side else 'front side'} detected)")

        # ---- Step 1: try to find a QR code in the SAME uploaded image ----
        batch, expiry_str, qr_found = get_batch_and_expiry_from_qr(path)

        # ---- Step 2: if no QR in the main image, ask for a separate QR photo ----
        if not qr_found:
            print("\nNo QR code detected in the uploaded image.")
            qr_path = input("If you have a separate photo of the QR code, enter its path (or press Enter to skip): ").strip()
            if qr_path and os.path.exists(qr_path):
                batch, expiry_str, qr_found = get_batch_and_expiry_from_qr(qr_path)

        # ---- Step 3: use QR-extracted batch/expiry if found, else fall back to OCR ----
        if qr_found and batch is not None:
            print(f"\n✅ Batch/Expiry extracted from QR code")
            expiry_status, expiry_conf = resolve_expiry_status_from_qr_string(expiry_str)
        else:
            print("\nFalling back to OCR extraction (no usable QR data found)")
            batch = extract_batch_number(path)
            expiry_status, expiry_conf = check_expiry_validity(path)

        print("Batch Number:", batch if batch else "Not detected")
        print("Expiry Status:", expiry_status)

        # ---- CNN prediction — unchanged ----
        result, conf = predict_counterfeit(model, img)
        final = decide(result, conf, batch, expiry_status, is_back_side)

        print("\n--- RESULT ---")
        print("Prediction:", result)
        print("Confidence:", round(conf, 2))
        print("Final:", final)

        run_grad_cam(img)