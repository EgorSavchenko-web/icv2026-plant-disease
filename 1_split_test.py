import os
import random
import shutil
from pathlib import Path

SRC = Path("./PlantVillage/train")
DST = Path("./PlantVillage/test")
PERCENT = 0.05
SEED = 42


def is_jpeg(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg"))


def main():
    if not SRC.is_dir():
        print(f"Folder not found: {SRC.resolve()}")
        return

    random.seed(SEED)
    total_moved = 0
    total_folders = 0

    for root, dirs, files in os.walk(SRC):
        root_path = Path(root)
        rel = root_path.relative_to(SRC)
        target_dir = DST / rel

        target_dir.mkdir(parents=True, exist_ok=True)
        total_folders += 1

        jpegs = [f for f in files if is_jpeg(f)]
        if not jpegs:
            continue

        n = max(1, round(len(jpegs) * PERCENT))
        chosen = random.sample(jpegs, n)

        for name in chosen:
            shutil.move(str(root_path / name), str(target_dir / name))

        total_moved += n
        print(f"{rel}: {n}/{len(jpegs)} files -> {target_dir}")

    print("\n" + "=" * 60)
    print("Done.")
    print(f"Folders processed: {total_folders}")
    print(f"Files moved: {total_moved}")
    print(f"Source: {SRC.resolve()}")
    print(f"Destination: {DST.resolve()}")


if __name__ == "__main__":
    main()