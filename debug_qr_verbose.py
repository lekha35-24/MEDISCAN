import cv2
import os
from qr_extractor import _preprocess_variants, _wechat_detector, qr_detector_classic

def diagnose(image_path):
    print(f"\n{'='*50}")
    print(f"Diagnosing: {image_path}")
    print('='*50)

    if not os.path.exists(image_path):
        print("❌ File does not exist at this path.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("❌ cv2.imread() failed — file exists but couldn't be read as an image.")
        print("   (Check the file isn't corrupted, and has a normal extension like .png/.jpg)")
        return

    h, w = img.shape[:2]
    print(f"✅ Image loaded successfully. Dimensions: {w}x{h}")

    if w < 100 or h < 100:
        print("⚠️  WARNING: Image is very small — QR detectors need reasonable resolution.")

    print(f"\nWeChat QR detector loaded: {_wechat_detector is not None}")
    if _wechat_detector is None:
        print("   -> WeChat detector unavailable, only classic detector will be tried.")

    variants = _preprocess_variants(img)
    variant_names = ["original", "upscaled 2x", "CLAHE enhanced", "adaptive threshold"]

    for name, variant in zip(variant_names, variants):
        print(f"\n--- Trying variant: {name} ---")

        if _wechat_detector is not None:
            try:
                results, points = _wechat_detector.detectAndDecode(variant)
                payloads = [r for r in results if r]
                if payloads:
                    print(f"  ✅ WeChat detector SUCCESS: {payloads}")
                    return
                else:
                    print(f"  ❌ WeChat detector: no QR found")
            except Exception as e:
                print(f"  ❌ WeChat detector error: {e}")

        try:
            retval, decoded_info, points, _ = qr_detector_classic.detectAndDecodeMulti(variant)
            if retval:
                payloads = [t for t in decoded_info if t]
                if payloads:
                    print(f"  ✅ Classic multi-detector SUCCESS: {payloads}")
                    return
                else:
                    print(f"  ❌ Classic multi-detector: QR-like pattern seen but couldn't decode text")
            else:
                print(f"  ❌ Classic multi-detector: no QR pattern found")
        except cv2.error as e:
            print(f"  ❌ Classic multi-detector error: {e}")

        try:
            data, points, _ = qr_detector_classic.detectAndDecode(variant)
            if data:
                print(f"  ✅ Classic single-detector SUCCESS: {data}")
                return
            else:
                print(f"  ❌ Classic single-detector: no QR found")
        except cv2.error as e:
            print(f"  ❌ Classic single-detector error: {e}")

    print("\n❌ FINAL RESULT: No detector/variant combination succeeded.")
    print("   Saving preprocessed variants to disk so you can visually inspect them...")

    os.makedirs("qr_debug_output", exist_ok=True)
    for name, variant in zip(variant_names, variants):
        safe_name = name.replace(" ", "_")
        out_path = os.path.join("qr_debug_output", f"{safe_name}.png")
        cv2.imwrite(out_path, variant)
    print(f"   Saved to ./qr_debug_output/ — open these and check if the QR is even visible/sharp.")


if __name__ == "__main__":
    path = input("Enter QR image path: ").strip()
    diagnose(path)