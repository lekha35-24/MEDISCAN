import os
import hashlib

def get_image_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_hash_dict(folder):
    hash_dict = {}
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            hash_dict[get_image_hash(path)] = path
    return hash_dict

train = get_hash_dict("dataset/train")
val = get_hash_dict("dataset/val")
test = get_hash_dict("dataset/test")

train_val_overlap = set(train.keys()) & set(val.keys())
train_test_overlap = set(train.keys()) & set(test.keys())
val_test_overlap = set(val.keys()) & set(test.keys())

print("\n--- Train vs Val Overlap ---")
if not train_val_overlap:
    print("None ✅")
for h in train_val_overlap:
    print("TRAIN:", train[h])
    print("VAL  :", val[h])

print("\n--- Train vs Test Overlap ---")
if not train_test_overlap:
    print("None ✅")
for h in train_test_overlap:
    print("TRAIN:", train[h])
    print("TEST :", test[h])

print("\n--- Val vs Test Overlap ---")
if not val_test_overlap:
    print("None ✅")
for h in val_test_overlap:
    print("VAL :", val[h])
    print("TEST:", test[h])

total_overlaps = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
print("\n" + "=" * 40)
if total_overlaps == 0:
    print("✅ No duplicate images across splits. Safe to train.")
else:
    print(f"⚠️ {total_overlaps} duplicate(s) found. Remove duplicates from val/test (keep only in train) before training.")