import os

def count_images(folder):
    count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                count += 1
    return count

folders = ["dataset/train", "dataset/val", "dataset/test"]

for f in folders:
    print(f"{f}: {count_images(f)} images")