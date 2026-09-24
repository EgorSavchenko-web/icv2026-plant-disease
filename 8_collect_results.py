from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd


TASK_ID_PATTERN = re.compile(r"task id=(?P<task_id>[0-9a-f]{32})")
WORKER_PATTERN = re.compile(r"(?P<worker>aiagent\d+:[a-z0-9,]+)")
DOCKER_PATTERN = re.compile(r"CLEARML_DOCKER_IMAGE=(?P<image>[^\s']+)")
CUDA_PATTERN = re.compile(r"CUDA Version (?P<cuda>[\d.]+)")
DEVICE_PATTERN = re.compile(r"Using device:\s*(?P<device>\w+)")
EVAL_DEVICE_PATTERN = re.compile(r"Device:\s*(?P<device>\w+)")
CLASSES_PATTERN = re.compile(r"Classes:\s*(?P<value>\d+)")
TRAIN_SAMPLES_PATTERN = re.compile(r"Train samples:\s*(?P<value>\d+)")
VAL_SAMPLES_PATTERN = re.compile(r"Validation samples:\s*(?P<value>\d+)")
TEST_SAMPLES_PATTERN = re.compile(r"Test samples:\s*(?P<value>\d+)")
TOTAL_PARAMS_PATTERN = re.compile(r"Total parameters:\s*(?P<value>\d+)")
TRAINABLE_PARAMS_PATTERN = re.compile(r"Trainable parameters:\s*(?P<value>\d+)")
EARLY_STOP_PATTERN = re.compile(r"Early stopping at epoch\s*(?P<value>\d+)")
EPOCH_PATTERN = re.compile(r"Epoch\s+(?P<epoch>\d+)/\d+.*?\|\s*(?P<seconds>[\d.]+)s")
FINAL_PATTERN = re.compile(
    r"Final test results\s*\|\s*accuracy\s+(?P<accuracy>[\d.]+)\s*\|\s*"
    r"macro F1\s+(?P<macro_f1>[\d.]+)\s*\|\s*best epoch\s+(?P<best_epoch>\d+)\s*\|\s*"
    r"training time\s+(?P<minutes>[\d.]+)\s*minutes"
)
DATASET_PATTERN = re.compile(r"(?P<name>[A-Za-z0-9_]+)\.(?P<dataset_id>[0-9a-f]{32})/artifacts")

TRAINING_OUTPUT_FILES = {
    "metrics.json": "training_metrics_{tag}.json",
    "classification_report.json": "classification_report_{tag}.json",
    "confusion_matrix.png": "confusion_matrix_{tag}.png",
    "training_curves.png": "training_curves_{tag}.png",
}


def search(pattern, text, group, cast=str):
    match = pattern.search(text)
    return cast(match.group(group)) if match else None


def parse_training_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

    epochs = [(int(m.group("epoch")), float(m.group("seconds"))) for m in EPOCH_PATTERN.finditer(text)]
    final = FINAL_PATTERN.search(text)

    record = {
        "log": path.as_posix(),
        "clearml_task_id": search(TASK_ID_PATTERN, text, "task_id"),
        "worker": search(WORKER_PATTERN, text, "worker"),
        "docker_image": search(DOCKER_PATTERN, text, "image"),
        "cuda_version": search(CUDA_PATTERN, text, "cuda"),
        "device": search(DEVICE_PATTERN, text, "device"),
        "num_classes": search(CLASSES_PATTERN, text, "value", int),
        "train_samples": search(TRAIN_SAMPLES_PATTERN, text, "value", int),
        "validation_samples": search(VAL_SAMPLES_PATTERN, text, "value", int),
        "test_samples": search(TEST_SAMPLES_PATTERN, text, "value", int),
        "total_parameters": search(TOTAL_PARAMS_PATTERN, text, "value", int),
        "trainable_parameters": search(TRAINABLE_PARAMS_PATTERN, text, "value", int),
        "early_stop_epoch": search(EARLY_STOP_PATTERN, text, "value", int),
        "epochs_completed": max((epoch for epoch, _ in epochs), default=None),
        "mean_epoch_seconds": round(sum(s for _, s in epochs) / len(epochs), 1) if epochs else None,
    }

    if final:
        record.update(
            {
                "test_accuracy": float(final.group("accuracy")),
                "test_macro_f1": float(final.group("macro_f1")),
                "best_epoch": int(final.group("best_epoch")),
                "training_time_minutes": float(final.group("minutes")),
            }
        )

    return record


def parse_evaluation_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

    datasets = sorted(
        {(m.group("name"), m.group("dataset_id")) for m in DATASET_PATTERN.finditer(text)}
    )

    return {
        "log": path.as_posix(),
        "clearml_task_id": search(TASK_ID_PATTERN, text, "task_id"),
        "worker": search(WORKER_PATTERN, text, "worker"),
        "docker_image": search(DOCKER_PATTERN, text, "image"),
        "cuda_version": search(CUDA_PATTERN, text, "cuda"),
        "device": search(EVAL_DEVICE_PATTERN, text, "device"),
        "clearml_datasets": [{"name": name, "id": dataset_id} for name, dataset_id in datasets],
    }


def build_provenance(logs_dir: Path, results_dir: Path):
    provenance = {"training": {}, "ablation_lr": {}, "evaluation": {}}

    for path in sorted((logs_dir / "training").glob("*.log")):
        provenance["training"][path.stem] = parse_training_log(path)

    for path in sorted((logs_dir / "ablation_lr").glob("*.log")):
        provenance["ablation_lr"][path.stem] = parse_training_log(path)

    for path in sorted((logs_dir / "evaluation").glob("*.log")):
        provenance["evaluation"][path.stem] = parse_evaluation_log(path)

    for metrics_path in sorted((results_dir / "clean").glob("metrics_*.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        name = f"inference_{payload['tag']}"
        entry = provenance["evaluation"].setdefault(name, {})
        entry["environment"] = payload.get("environment")
        entry["latency"] = payload.get("latency")
        entry["peak_gpu_memory_mb"] = payload.get("peak_gpu_memory_mb")

    (results_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    return provenance


def import_training_outputs(outputs_dir: Path, clean_dir: Path):
    imported = []

    if not outputs_dir.is_dir():
        return imported

    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            continue

        tag = json.loads(metrics_path.read_text(encoding="utf-8")).get("mode", run_dir.name)

        for source_name, target_pattern in TRAINING_OUTPUT_FILES.items():
            source = run_dir / source_name
            if source.is_file():
                shutil.copy2(source, clean_dir / target_pattern.format(tag=tag))
                imported.append(target_pattern.format(tag=tag))

        failure_source = run_dir / "failure_cases"
        if failure_source.is_dir():
            failure_target = clean_dir / f"failure_cases_{tag}"
            failure_target.mkdir(parents=True, exist_ok=True)
            for image in sorted(failure_source.glob("*")):
                if image.is_file():
                    shutil.copy2(image, failure_target / image.name)
                    imported.append(f"failure_cases_{tag}/{image.name}")

    return imported


def build_main_table(results_dir: Path, provenance):
    rows = []

    for metrics_path in sorted((results_dir / "clean").glob("metrics_*.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        tag = payload["tag"]
        training = provenance["training"].get(tag, {})

        row = {
            "model": tag,
            "trainable_parameters": training.get("trainable_parameters"),
            "total_parameters": training.get("total_parameters"),
            "epochs_completed": training.get("epochs_completed"),
            "best_epoch": training.get("best_epoch"),
            "training_time_minutes": training.get("training_time_minutes"),
            "test_accuracy": payload["accuracy"],
            "test_macro_f1": payload["macro_f1"],
            "errors": payload["errors"],
            "mean_confidence_correct": payload.get("mean_confidence_correct"),
            "mean_confidence_incorrect": payload.get("mean_confidence_incorrect"),
            "inference_device": payload.get("environment", {}).get("device_type"),
            "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb"),
        }

        if payload.get("environment", {}).get("device_type") == "cuda":
            for entry in payload.get("latency", []):
                row[f"latency_ms_image_batch{entry['batch_size']}"] = round(
                    entry["mean_ms_per_image"], 3
                )

        rows.append(row)

    if not rows:
        return None

    frame = pd.DataFrame(rows).sort_values("model")
    frame.to_csv(results_dir / "main_table.csv", index=False, encoding="utf-8")

    return frame


def build_inference_cost(results_dir: Path):
    for metrics_path in sorted((results_dir / "clean").glob("metrics_*.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        environment = payload.get("environment", {})

        if environment.get("device_type") != "cuda":
            continue

        cost = {
            "note": (
                "Both conditions share the same ResNet-50 graph and the same 23,585,894 parameters, "
                "so the forward pass and therefore the inference cost are identical. Latency and "
                "memory are a property of the architecture and are measured once."
            ),
            "measured_from": payload["tag"],
            "device": environment.get("gpu_name"),
            "cuda": environment.get("cuda"),
            "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb"),
            "full_pass_seconds": payload.get("full_pass_seconds"),
            "samples": payload.get("samples"),
        }

        for entry in payload.get("latency", []):
            cost[f"latency_ms_per_image_batch{entry['batch_size']}"] = round(
                entry["mean_ms_per_image"], 3
            )
            cost[f"throughput_images_per_second_batch{entry['batch_size']}"] = round(
                entry["throughput_images_per_second"], 1
            )

        return cost

    return None


def build_final_metrics(results_dir: Path, provenance):
    payload = {"clean": {}, "training": provenance["training"], "ablation_lr": {}}

    for metrics_path in sorted((results_dir / "clean").glob("metrics_*.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["clean"][data["tag"]] = {
            "accuracy": data["accuracy"],
            "macro_f1": data["macro_f1"],
            "weighted_f1": data.get("weighted_f1"),
            "errors": data["errors"],
            "latency": data.get("latency"),
            "peak_gpu_memory_mb": data.get("peak_gpu_memory_mb"),
            "environment": data.get("environment"),
        }

    for name, record in provenance["ablation_lr"].items():
        payload["ablation_lr"][name] = {
            "test_accuracy": record.get("test_accuracy"),
            "test_macro_f1": record.get("test_macro_f1"),
            "best_epoch": record.get("best_epoch"),
            "epochs_completed": record.get("epochs_completed"),
            "training_time_minutes": record.get("training_time_minutes"),
        }

    inference_cost = build_inference_cost(results_dir)
    if inference_cost:
        payload["inference_cost"] = inference_cost

    robustness_path = results_dir / "robustness" / "robustness_metrics.csv"
    if robustness_path.is_file():
        payload["robustness"] = pd.read_csv(robustness_path).to_dict("records")

    failure_path = results_dir / "failures" / "failure_summary.json"
    if failure_path.is_file():
        payload["failures"] = json.loads(failure_path.read_text(encoding="utf-8"))

    (results_dir / "final_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Assemble the final results tree, the main table and the run provenance record."
    )
    parser.add_argument("--logs-dir", default="./logs")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--training-outputs", default="./outputs")
    return parser


def main():
    args = build_parser().parse_args()

    logs_dir = Path(args.logs_dir)
    results_dir = Path(args.results_dir)
    clean_dir = results_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    imported = import_training_outputs(Path(args.training_outputs), clean_dir)
    if imported:
        print(f"Imported {len(imported)} file(s) from local training outputs")

    provenance = build_provenance(logs_dir, results_dir)
    print(
        f"Provenance: {len(provenance['training'])} training run(s), "
        f"{len(provenance['ablation_lr'])} ablation run(s), "
        f"{len(provenance['evaluation'])} evaluation run(s)"
    )

    frame = build_main_table(results_dir, provenance)

    if frame is None:
        print("[WARN] No inference metrics found under results/clean. Run 5_inference.py first.")
    else:
        print("\n=== Main table ===")
        print(frame.to_string(index=False))

    build_final_metrics(results_dir, provenance)
    print(f"\nResults tree assembled at {results_dir.resolve()}")


if __name__ == "__main__":
    main()
