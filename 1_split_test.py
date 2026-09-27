import argparse
import os
import random
import shutil
from pathlib import Path

SRC = Path("./PlantVillage/train")
DST = Path("./PlantVillage/test")
MANIFEST = Path("./test_split_manifest.txt")
PERCENT = 0.05
SEED = 42


def is_jpeg(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg"))


def sanitize(relative: Path) -> str:
    return "/".join(part.replace(" ", "_") for part in relative.parts)


def split_from_manifest(manifest_path: Path):
    wanted = {line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    moved = set()
    for root, dirs, files in os.walk(SRC):
        dirs.sort()
        root_path = Path(root)
        rel = root_path.relative_to(SRC)
        for name in sorted(files):
            key = sanitize(rel / name)
            if key not in wanted:
                continue
            target_dir = DST / rel
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root_path / name), str(target_dir / name))
            moved.add(key)
    missing = sorted(wanted - moved)
    print(f"Moved {len(moved)} of {len(wanted)} files listed in {manifest_path}")
    if missing:
        print(f"{len(missing)} listed files were not found in {SRC}, first ones:")
        for entry in missing[:10]:
            print(f"  {entry}")
        raise SystemExit(1)


def split_at_random():
    random.seed(SEED)
    total_moved = 0
    manifest = []
    for root, dirs, files in os.walk(SRC):
        dirs.sort()
        root_path = Path(root)
        rel = root_path.relative_to(SRC)
        jpegs = sorted(f for f in files if is_jpeg(f))
        if not jpegs:
            continue
        target_dir = DST / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        n = max(1, round(len(jpegs) * PERCENT))
        chosen = random.sample(jpegs, n)
        for name in sorted(chosen):
            shutil.move(str(root_path / name), str(target_dir / name))
            manifest.append(sanitize(rel / name))
        total_moved += n
        print(f"{rel}: {n}/{len(jpegs)} files -> {target_dir}")
    out = Path("./test_split_manifest.new.txt")
    out.write_text("\n".join(sorted(manifest)) + "\n", encoding="utf-8")
    print(f"Moved {total_moved} files. Manifest of this new split written to {out}")


def main():
    parser = argparse.ArgumentParser(description="Carve the test split out of PlantVillage/train")
    parser.add_argument("--manifest", type=Path, default=MANIFEST,
                        help="move exactly the files listed here (default: the committed manifest)")
    parser.add_argument("--random", action="store_true",
                        help="draw a new 5%% split with seed 42 instead of reproducing the committed one")
    args = parser.parse_args()

    if not SRC.is_dir():
        raise SystemExit(f"Folder not found: {SRC.resolve()}")
    if args.random:
        split_at_random()
    else:
        if not args.manifest.is_file():
            raise SystemExit(f"Manifest not found: {args.manifest.resolve()}")
        split_from_manifest(args.manifest)
    print(f"Source: {SRC.resolve()}")
    print(f"Destination: {DST.resolve()}")


if __name__ == "__main__":
    main()
