# merge_pool_into_dataset_grouped.py
import os
import shutil
import random
from collections import defaultdict

random.seed(42)

POOL_DIR = "real_world_train_pool/real"   # your video frames + friend photos live here
TRAIN_DEST = "dataset/train/Real"
VAL_DEST = "dataset/val/Real"

# Group files by their "source" — assumes video frames are named like
# video_frame_0.jpg, video_frame_1.jpg from ONE video, or vid2_frame_0.jpg
# from a different one. Adjust the grouping key logic below to match
# your actual filenames from extract_video.py.

def get_source_group(filename):
    # Example: "video_frame_23.jpg" -> group key "video"
    # If you extracted multiple videos with different prefixes
    # (e.g. medA_frame_0.jpg, medB_frame_0.jpg), this naturally groups them.
    name = filename.rsplit("_frame_", 1)[0] if "_frame_" in filename else filename
    return name

groups = defaultdict(list)
for f in os.listdir(POOL_DIR):
    groups[get_source_group(f)].append(f)

group_keys = list(groups.keys())
random.shuffle(group_keys)

split_idx = int(len(group_keys) * 0.85)
train_groups = group_keys[:split_idx]
val_groups = group_keys[split_idx:]

train_count, val_count = 0, 0

for g in train_groups:
    for f in groups[g]:
        shutil.copy(os.path.join(POOL_DIR, f), os.path.join(TRAIN_DEST, f))
        train_count += 1

for g in val_groups:
    for f in groups[g]:
        shutil.copy(os.path.join(POOL_DIR, f), os.path.join(VAL_DEST, f))
        val_count += 1

print(f"Groups: {len(group_keys)} total, {len(train_groups)} train, {len(val_groups)} val")
print(f"Files: {train_count} train, {val_count} val")