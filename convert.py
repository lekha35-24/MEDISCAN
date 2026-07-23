from PIL import Image
import os

folder = "real_world_test"

for label in ["real", "fake"]:
    path = os.path.join(folder, label)

    for img_name in os.listdir(path):
        if img_name.endswith(".webp"):
            img_path = os.path.join(path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")

                new_name = img_name.replace(".webp", ".jpg")
                new_path = os.path.join(path, new_name)

                img.save(new_path, "JPEG")

                os.remove(img_path)  # remove old file

                print(f"✅ Converted: {img_name} → {new_name}")

            except Exception as e:
                print(f"❌ Error: {img_name} - {e}")

print("🎯 Conversion complete!")