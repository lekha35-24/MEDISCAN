import cv2
import os

VIDEO_DIR = "videos"
OUTPUT_DIR = "real_world_test/real"
FRAME_INTERVAL = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

total_saved = 0

for video_file in os.listdir(VIDEO_DIR):
    if not video_file.endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEO_DIR, video_file)
    cap = cv2.VideoCapture(video_path)

    count = 0
    saved = 0

    print(f"Processing {video_file}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_INTERVAL == 0:
            out_name = f"{video_file.split('.')[0]}_{saved}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, frame)
            saved += 1

        count += 1

    cap.release()
    total_saved += saved
    print(f"Saved {saved} frames from {video_file}")

print(f"\nTOTAL frames saved: {total_saved}")