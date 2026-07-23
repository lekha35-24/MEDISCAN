from PIL import Image
import os

folder = "dataset"
splits = ["train", "val", "test"]
classes = ["Fake", "Real"]
valid_ext = (".jpg", ".jpeg", ".png")

total_images = 0
invalid_files = 0
corrupt_files = 0
small_images = 0

for split in splits:
    for cls in classes:
        path = os.path.join(folder, split, cls)
        print(f"\n📂 Checking folder: {path}")

        if not os.path.exists(path):
            print(f"❌ Folder missing: {path}")
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            total_images += 1

            if not img_name.lower().endswith(valid_ext):
                print(f"❌ Invalid format: {img_name}")
                invalid_files += 1
                continue

            try:
                img = Image.open(img_path)
                img.verify()
            except Exception:
                print(f"❌ Corrupted image: {img_name}")
                corrupt_files += 1
                continue

            try:
                img = Image.open(img_path)
                width, height = img.size
                if width < 50 or height < 50:
                    print(f"⚠️ Too small: {img_name} ({width}x{height})")
                    small_images += 1
            except Exception:
                print(f"❌ Error reading size: {img_name}")
                corrupt_files += 1

print("\n📊 ===== SUMMARY =====")
print(f"Total images checked: {total_images}")
print(f"Invalid format: {invalid_files}")
print(f"Corrupted images: {corrupt_files}")
print(f"Too small images: {small_images}")
print("✅ Verification complete!")