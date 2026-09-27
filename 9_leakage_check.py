import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_images(root: Path):
    return sorted(p.as_posix() for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)


def dhash(path: str) -> int:
    image = Image.open(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.int16)
    bits = (array[:, 1:] > array[:, :-1]).flatten()
    return int(np.packbits(bits).view(np.uint64)[0])


def hash_split(data_dir: Path, split: str, state_dir: Path, budget: float):
    root = data_dir / split
    if not root.exists():
        raise SystemExit(f"missing split directory: {root}")
    state_path = state_dir / f"{split}.json"
    hashes = json.loads(state_path.read_text()) if state_path.exists() else {}
    paths = list_images(root)
    todo = [p for p in paths if p not in hashes]
    start = time.perf_counter()
    for path in todo:
        if budget > 0 and time.perf_counter() - start > budget:
            break
        try:
            hashes[path] = dhash(path)
        except Exception:
            hashes[path] = None
    state_path.write_text(json.dumps(hashes))
    remaining = len(paths) - len(hashes)
    print(f"{split}: {len(hashes)}/{len(paths)} hashed, {remaining} remaining")
    return hashes, remaining


def readable(hashes: dict):
    return {p: v for p, v in hashes.items() if v is not None and v >= 0}


def nearest_neighbours(test_hashes: dict, train_hashes: dict):
    test_hashes = readable(test_hashes)
    train_hashes = readable(train_hashes)
    train_paths = np.array(sorted(train_hashes))
    train_values = np.array([train_hashes[p] for p in train_paths], dtype=np.uint64)
    train_bits = np.unpackbits(train_values.view(np.uint8).reshape(-1, 8), axis=1)
    records = {}
    for test_path in sorted(test_hashes):
        value = np.uint64(test_hashes[test_path])
        bits = np.unpackbits(np.array([value], dtype=np.uint64).view(np.uint8))
        distances = np.count_nonzero(train_bits != bits, axis=1)
        index = int(np.argmin(distances))
        records[test_path] = {
            "distance": int(distances[index]),
            "nearest_train": str(train_paths[index]),
        }
    return records


def class_of(path: str) -> str:
    return Path(path).parent.name


def accuracy_shift(predictions_path: Path, flagged: set):
    if not predictions_path.exists():
        return {}
    frame = pd.read_csv(predictions_path)
    frame = frame[(frame["corruption"] == "none")].copy()
    if frame.empty:
        return {}
    frame["key"] = frame["path"].map(lambda p: "/".join(Path(p).parts[-2:]))
    shifts = {}
    for model, group in frame.groupby("model"):
        keep = group[~group["key"].isin(flagged)]
        full = float(group["correct"].mean())
        pruned = float(keep["correct"].mean()) if len(keep) else float("nan")
        shifts[str(model)] = {
            "accuracy_all": round(full, 6),
            "accuracy_without_near_duplicates": round(pruned, 6),
            "delta": round(pruned - full, 6),
            "removed": int(len(group) - len(keep)),
            "errors_on_near_duplicates": int((~group[group["key"].isin(flagged)]["correct"]).sum()),
            "min_distance_among_errors": None,
        }
    return shifts


def main():
    parser = argparse.ArgumentParser(description="Perceptual-hash leakage check between train and test splits")
    parser.add_argument("--data-dir", type=Path, default=Path("PlantVillage"))
    parser.add_argument("--state-dir", type=Path, default=Path("results/leakage/hashes"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/leakage"))
    parser.add_argument("--predictions", type=Path, default=Path("results/robustness/predictions_robustness.csv"))
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--budget", type=float, default=0.0,
                        help="seconds of hashing per invocation; 0 hashes everything in one run")
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args()

    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarise_only:
        train_hashes = json.loads((args.state_dir / "train.json").read_text())
        test_hashes = json.loads((args.state_dir / "test.json").read_text())
        remaining = 0
    else:
        train_hashes, train_left = hash_split(args.data_dir, "train", args.state_dir, args.budget)
        test_hashes, test_left = hash_split(args.data_dir, "test", args.state_dir, args.budget)
        remaining = train_left + test_left
    if remaining:
        print("hashing incomplete, rerun this script to continue")
        return

    neighbours_path = args.out_dir / "nearest_neighbours.json"
    if neighbours_path.exists() and args.summarise_only:
        records = json.loads(neighbours_path.read_text())
    else:
        records = nearest_neighbours(test_hashes, train_hashes)
        neighbours_path.write_text(json.dumps(records, indent=2))

    distances = np.array([r["distance"] for r in records.values()])
    flagged = {}
    for path, record in records.items():
        if record["distance"] <= args.threshold:
            flagged[path] = record
    same_class = sum(1 for p, r in flagged.items() if class_of(p) == class_of(r["nearest_train"]))

    rows = []
    for path, record in sorted(flagged.items()):
        rows.append({
            "test_image": path,
            "nearest_train_image": record["nearest_train"],
            "hamming_distance": record["distance"],
            "same_class": class_of(path) == class_of(record["nearest_train"]),
        })
    pd.DataFrame(rows).to_csv(args.out_dir / "near_duplicates.csv", index=False)

    flagged_keys = {"/".join(Path(p).parts[-2:]) for p in flagged}
    shifts = accuracy_shift(args.predictions, flagged_keys)

    error_distances = {}
    if args.predictions.exists():
        frame = pd.read_csv(args.predictions)
        frame = frame[frame["corruption"] == "none"]
        lookup = {"/".join(Path(p).parts[-2:]): r["distance"] for p, r in records.items()}
        for model, group in frame.groupby("model"):
            errors = group[~group["correct"]]
            keys = ["/".join(Path(p).parts[-2:]) for p in errors["path"]]
            values = [lookup[k] for k in keys if k in lookup]
            error_distances[str(model)] = int(min(values)) if values else None
    for model, value in error_distances.items():
        if model in shifts:
            shifts[model]["min_distance_among_errors"] = value

    summary = {
        "test_images": int(len(records)),
        "train_images": int(len(train_hashes)),
        "unreadable_images": int(sum(1 for h in (train_hashes, test_hashes)
                                     for v in h.values() if v is None or v < 0)),
        "orientation_sensitive": True,
        "threshold": args.threshold,
        "exact_matches": int((distances == 0).sum()),
        "within_threshold": int(len(flagged)),
        "within_threshold_fraction": round(float(len(flagged)) / len(records), 6),
        "within_threshold_same_class": int(same_class),
        "distance_median": float(np.median(distances)),
        "distance_mean": round(float(distances.mean()), 4),
        "distance_min": int(distances.min()),
        "per_model": shifts,
    }
    (args.out_dir / "leakage_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
