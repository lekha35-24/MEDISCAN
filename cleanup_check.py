import os

# Folders to scan for leftover synthetic-fake files
TARGETS = [
    "dataset/train/Fake",
    "dataset/val/Fake",
    "dataset/test/Fake",
]

# Prefixes produced by the old deleted scripts (gen_fake_data.py, genfake.py, genfake1.py, augment_real.py)
SUSPICIOUS_PREFIXES = ("fake_", "blur_", "noise_", "rot_", "bright_", "flip_")

DRY_RUN = True  # set to False only after you've reviewed the printed list

def scan_and_clean():
    total_found = 0
    total_deleted = 0

    for folder in TARGETS:
        if not os.path.exists(folder):
            print(f"⏭️  Skipping (not found): {folder}")
            continue

        files = os.listdir(folder)
        suspicious = [f for f in files if f.startswith(SUSPICIOUS_PREFIXES)]

        if not suspicious:
            print(f"✅ {folder} — clean, no synthetic files found ({len(files)} total images)")
            continue

        total_found += len(suspicious)
        print(f"\n⚠️  {folder} — found {len(suspicious)} suspicious file(s):")
        for f in suspicious[:15]:
            print("   -", f)
        if len(suspicious) > 15:
            print(f"   ...and {len(suspicious) - 15} more")

        if not DRY_RUN:
            for f in suspicious:
                os.remove(os.path.join(folder, f))
                total_deleted += 1
            print(f"   🗑️  Deleted {len(suspicious)} file(s) from {folder}")

    print("\n" + "=" * 40)
    if DRY_RUN:
        print(f"DRY RUN complete. {total_found} suspicious file(s) found across all folders.")
        print("Review the list above. If it looks correct, set DRY_RUN = False and re-run to actually delete.")
    else:
        print(f"Cleanup complete. {total_deleted} file(s) deleted.")

if __name__ == "__main__":
    scan_and_clean()