import os

folder = "dataset/train/Fake"

deleted = 0

for file in os.listdir(folder):

    if file.startswith("fake_"):
        os.remove(os.path.join(folder, file))
        deleted += 1

print("Deleted generated fake images:", deleted)