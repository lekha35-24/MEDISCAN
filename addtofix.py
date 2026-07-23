# add_to_fixed_test.py
import os
import shutil

NEW_PHOTOS_DIR = "new_friend_photos"       # drop new photos here first
FIXED_TEST_DIR = "real_world_test_fixed/real"

os.makedirs(FIXED_TEST_DIR, exist_ok=True)

count = 0
for f in os.listdir(NEW_PHOTOS_DIR):
    shutil.copy(os.path.join(NEW_PHOTOS_DIR, f), os.path.join(FIXED_TEST_DIR, f))
    count += 1

print(f"Added {count} new images to the fixed real-world test set.")
print("IMPORTANT: do not use these for training.")