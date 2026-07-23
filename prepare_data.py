import os
import shutil
import random

random.seed(42)  # reproducibility

fake_dir = "raw/Fake"
real_dir = "raw/Real"

train_dir = "dataset/train"
val_dir = "dataset/val"

for split in ["train", "val"]:
    for cls in ["Fake", "Real"]:
        os.makedirs(os.path.join("dataset", split, cls), exist_ok=True)

def split_data(source, train_dest, val_dest, split_ratio=0.8):
    files = os.listdir(source)
    random.shuffle(files)

    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]

    for f in train_files:
        shutil.copy(os.path.join(source, f), os.path.join(train_dest, f))

    for f in val_files:
        shutil.copy(os.path.join(source, f), os.path.join(val_dest, f))

split_data(fake_dir, os.path.join(train_dir, "Fake"), os.path.join(val_dir, "Fake"))
split_data(real_dir, os.path.join(train_dir, "Real"), os.path.join(val_dir, "Real"))

print("Dataset split completed correctly")