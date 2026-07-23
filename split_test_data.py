import os
import shutil
import random

random.seed(42)

train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

classes = ["Fake", "Real"]

for cls in classes:
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

def move_images(source_dir, class_name, num_images):
    folder = os.path.join(source_dir, class_name)
    images = os.listdir(folder)
    random.shuffle(images)
    selected = images[:num_images]

    for img in selected:
        src = os.path.join(folder, img)
        dst = os.path.join(test_dir, class_name, img)
        shutil.move(src, dst)

for cls in classes:
    move_images(train_dir, cls, 20)

for cls in classes:
    move_images(val_dir, cls, 10)

print("✅ Images moved successfully to test folder")