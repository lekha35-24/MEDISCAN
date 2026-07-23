# undo_realworld_merge.py
import os

DATASET_DIRS = ["dataset/train/Real", "dataset/val/Real"]
SOURCE_DIRS = ["real_world_test_pool/real", "real_world_test/real"]

# Build a set of filenames that came from your real-world sources
source_filenames = set()
for src in SOURCE_DIRS:
    if os.path.exists(src):
        source_filenames.update(os.listdir(src))

removed = 0
for d in DATASET_DIRS:
    if not os.path.exists(d):
        continue
    for f in os.listdir(d):
        if f in source_filenames:
            os.remove(os.path.join(d, f))
            removed += 1

print(f"Removed {removed} previously-merged real-world files from dataset/train and val.")
print("Kaggle images were not touched.")