# debug_qr.py
import sys
from qr_extractor import decode_qr_from_image

if __name__ == "__main__":
    path = input("Enter QR image path: ").strip()
    payloads = decode_qr_from_image(path)

    if not payloads:
        print("❌ No QR code decoded.")
    else:
        print(f"✅ Decoded {len(payloads)} QR payload(s):")
        for i, p in enumerate(payloads):
            print(f"  [{i+1}] {p}")