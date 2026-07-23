# reorganize_folders.py
import os
import shutil

# Your ORIGINAL 5 camera photos = permanent, never-touched test set
FIXED_TEST_FILES = [
    "real1.jpeg",
    "real2.jpeg",
    "WhatsApp Image 2026-07-14 at 8.09.41 PM.jpeg",
    "WhatsApp Image 2026-07-14 at 8.09.42 PM (1).jpeg",
    "WhatsApp Image 2026-07-14 at 8.09.42 PM.jpeg",
]

SRC = "real_world_test/real"
FIXED_TEST_DIR = "real_world_test_FIXED/real"   # permanent holdout — never grows
TRAIN_POOL_DIR = "real_world_train_pool/real"    # new photos go here instead

os.makedirs(FIXED_TEST_DIR, exist_ok=True)
os.makedirs(TRAIN_POOL_DIR, exist_ok=True)

for fname in os.listdir(SRC):
    src_path = os.path.join(SRC, fname)
    if fname in FIXED_TEST_FILES:
        shutil.copy(src_path, os.path.join(FIXED_TEST_DIR, fname))
    else:
        shutil.copy(src_path, os.path.join(TRAIN_POOL_DIR, fname))

print("Done. Fixed test set and training pool separated.")