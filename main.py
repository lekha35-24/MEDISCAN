import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from counterfeit_model import predict_counterfeit, load_model, grad_cam, transform, device
from batch_ocr import extract_batch_number, get_text_density
from expiry_checker import check_expiry_validity

model = load_model()

HIGH_CONF = 0.90
MED_CONF = 0.80
TEXT_DENSITY_THRESHOLD = 25


def decide(result, conf, batch, expiry_status, is_back_side):
    if is_back_side:
        if batch is None and expiry_status == "NOT FOUND":
            trust_note = " (OCR unreadable — relying on visual model)"
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
            return "SUSPICIOUS (visual only, no OCR verification)"
        if result == "Real" and conf > HIGH_CONF:
            return "LIKELY GENUINE (visual only, no OCR verification)"
        return "UNCERTAIN (visual only — consider photographing back of strip for OCR verification)"


def run_grad_cam(img):
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

        batch = extract_batch_number(path)
        print("Batch Number:", batch if batch else "Not detected")

        expiry_status, expiry_conf = check_expiry_validity(path)
        print("Expiry Status:", expiry_status)

        result, conf = predict_counterfeit(model, img)
        final = decide(result, conf, batch, expiry_status, is_back_side)

        print("\n--- RESULT ---")
        print("Prediction:", result)
        print("Confidence:", round(conf, 2))
        print("Final:", final)

        run_grad_cam(img)