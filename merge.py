# merge_realworld_grouped.py
import os
import re
import shutil
import random
from collections import defaultdict

random.seed(42)

SOURCE_DIRS = ["real_world_test_pool/real", "real_world_test/real"]
TRAIN_DEST = "dataset/train/Real"
VAL_DEST = "dataset/val/Real"

def get_source_group(filename):
    """
    Groups video frames from the same source video together.
    'med1_0.jpg', 'med1_1.jpg' -> group 'med1' (same physical medicine, must stay together)
    'WhatsApp Image ....jpeg'  -> group is the whole filename (no trailing _<number> to strip,
                                   so each friend photo is its own independent group)
    """
    name = os.path.splitext(filename)[0]
    stripped = re.sub(r'_\d+$', '', name)
    return stripped

groups = defaultdict(list)  # group_key -> list of (filename, full_source_path)

for src in SOURCE_DIRS:
    if not os.path.exists(src):
        continue
    for f in os.listdir(src):
        group_key = get_source_group(f)
        groups[group_key].append((f, os.path.join(src, f)))

group_keys = list(groups.keys())
random.shuffle(group_keys)

split_idx = int(len(group_keys) * 0.85)
train_groups = group_keys[:split_idx]
val_groups = group_keys[split_idx:]

train_count, val_count = 0, 0

for g in train_groups:
    for fname, path in groups[g]:
        shutil.copy(path, os.path.join(TRAIN_DEST, fname))
        train_count += 1

for g in val_groups:
    for fname, path in groups[g]:
        shutil.copy(path, os.path.join(VAL_DEST, fname))
        val_count += 1

print(f"Total groups (distinct medicines/photos): {len(group_keys)}")
print(f"  -> {len(train_groups)} groups to train ({train_count} images)")
print(f"  -> {len(val_groups)} groups to val ({val_count} images)")
print("\nGroup breakdown:")
for g in group_keys:
    dest = "TRAIN" if g in train_groups else "VAL"
    print(f"  {g}: {len(groups[g])} images -> {dest}")