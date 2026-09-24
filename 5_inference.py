from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


PROJECT_NAME = os.environ.get("PLANTVILLAGE_CLEARML_PROJECT", "ICV2026_PlantDisease")
QUEUE_NAME = os.environ.get("PLANTVILLAGE_CLEARML_QUEUE", "")
DATASET_ID = os.environ.get("PLANTVILLAGE_DATASET_ID", "687ee9f1c8dd4af98240e55e4b18258a")
FALLBACK_MEAN = [0.485, 0.456, 0.406]
FALLBACK_STD = [0.229, 0.224, 0.225]


def get_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return device


def load_bundle(checkpoint_path: Path, device):
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    required = ("classes", "model_state_dict")
    missing = [key for key in required if key not in bundle]
    if missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {missing}")

    classes = list(bundle["classes"])

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()

    return model, bundle, classes


def build_eval_transform(bundle):
    image_size = int(bundle.get("image_size", 224))
    mean = bundle.get("imagenet_mean", FALLBACK_MEAN)
    std = bundle.get("imagenet_std", FALLBACK_STD)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transform, image_size


def describe_environment(device):
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device_type": device.type,
    }

    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpu_total_memory_gb"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1024**3, 1
        )

    return info


def predict_single(model, transform, classes, image_path: Path, device, topk: int):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    order = np.argsort(probabilities)[::-1][: min(topk, len(classes))]

    return [(classes[int(index)], float(probabilities[int(index)])) for index in order]


def build_split_dataset(data_root: Path, split: str, transform, limit: int | None):
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    dataset = ImageFolder(split_dir, transform=transform)
    sample_paths = [path for path, _ in dataset.samples]

    if limit is not None and limit < len(dataset):
        indices = list(range(limit))
        return Subset(dataset, indices), dataset.classes, sample_paths[:limit]

    return dataset, dataset.classes, sample_paths


def run_split(model, loader, device):
    all_targets = []
    all_predictions = []
    all_confidences = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            probabilities = torch.softmax(model(inputs), dim=1)
            confidences, predictions = probabilities.max(dim=1)

            all_targets.append(targets)
            all_predictions.append(predictions.cpu())
            all_confidences.append(confidences.cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    wall_seconds = time.perf_counter() - start

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)

    return (
        torch.cat(all_targets).numpy(),
        torch.cat(all_predictions).numpy(),
        torch.cat(all_confidences).numpy(),
        wall_seconds,
        peak_memory_mb,
    )


def build_latency_pool(sample_paths, transform, size: int):
    tensors = [transform(Image.open(path).convert("RGB")) for path in sample_paths[:size]]
    if not tensors:
        raise RuntimeError("No images available for latency measurement.")
    return torch.stack(tensors)


def measure_latency(model, device, pool, batch_size: int, warmup: int, iterations: int):
    if pool.size(0) < batch_size:
        repeats = (batch_size + pool.size(0) - 1) // pool.size(0)
        pool = pool.repeat(repeats, 1, 1, 1)

    batch = pool[:batch_size].to(device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    durations = []

    with torch.inference_mode():
        for _ in range(iterations):
            start = time.perf_counter()
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append((time.perf_counter() - start) * 1000.0)

    durations = np.array(durations)

    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "mean_ms_per_batch": float(durations.mean()),
        "median_ms_per_batch": float(np.median(durations)),
        "p95_ms_per_batch": float(np.percentile(durations, 95)),
        "mean_ms_per_image": float(durations.mean() / batch_size),
        "throughput_images_per_second": float(batch_size * 1000.0 / durations.mean()),
    }


def write_outputs(output_dir: Path, tag: str, sample_paths, classes, targets, predictions, confidences, summary):
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_frame = pd.DataFrame(
        {
            "path": [Path(path).as_posix() for path in sample_paths],
            "true_class": [classes[int(index)] for index in targets],
            "predicted_class": [classes[int(index)] for index in predictions],
            "confidence": confidences,
            "correct": targets == predictions,
        }
    )
    predictions_frame.to_csv(output_dir / f"predictions_{tag}.csv", index=False, encoding="utf-8")

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )

    per_class_rows = [
        {
            "class": name,
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": int(report[name]["support"]),
        }
        for name in classes
    ]
    per_class_frame = pd.DataFrame(per_class_rows).sort_values("f1")
    per_class_frame.to_csv(output_dir / f"per_class_metrics_{tag}.csv", index=False, encoding="utf-8")

    (output_dir / f"metrics_{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return predictions_frame, per_class_frame


def running_on_agent():
    return bool(os.environ.get("CLEARML_TASK_ID"))


def transfer_arguments(task, raw_arguments, parser, args):
    payload = {
        "argv": ""
        if running_on_agent()
        else " ".join(shlex.quote(argument) for argument in raw_arguments)
    }

    task.connect(payload, name="Args")

    if not running_on_agent():
        return args

    if not payload["argv"]:
        raise RuntimeError("No command line arguments were stored in the ClearML task.")

    return parser.parse_args(shlex.split(payload["argv"]))


def start_clearml(task_name, raw_arguments, parser, args):
    from clearml import Dataset, Task

    task = Task.init(project_name=PROJECT_NAME, task_name=task_name)
    args = transfer_arguments(task, raw_arguments, parser, args)

    if not running_on_agent():
        os.environ["CLEARML_APT_INSTALL"] = ""
        task.set_packages(
            [
                "clearml>=2.1",
                "boto3",
                "numpy>=1.26",
                "pandas>=2.0",
                "scikit-learn>=1.4",
                "matplotlib>=3.8",
                "torch==2.10.0",
                "torchvision==0.25.0",
            ]
        )
        task.execute_remotely(queue_name=args.queue, exit_process=True)

    data_dir = Path(Dataset.get(dataset_id=DATASET_ID, alias="PlantVillage").get_local_copy())

    return task, args, data_dir


def find_checkpoint_in_dir(local_dir: Path, filename: str | None) -> Path:
    if filename:
        candidate = local_dir / filename
        if candidate.is_file():
            return candidate
        matches = [p for p in local_dir.rglob(filename) if p.is_file()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"File '{filename}' not found in dataset at {local_dir}"
            )
        raise RuntimeError(
            f"Multiple files named '{filename}' found in dataset: {matches}"
        )

    candidates = sorted(p for p in local_dir.rglob("*.pth") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No .pth files found in dataset at {local_dir}")
    if len(candidates) > 1:
        names = ", ".join(p.relative_to(local_dir).as_posix() for p in candidates)
        raise RuntimeError(
            f"Multiple .pth files found in dataset: {names}. "
            f"Pass --model-filename to disambiguate."
        )
    return candidates[0]


def resolve_checkpoint(args) -> Path:
    requested = [
        ("--checkpoint", args.checkpoint),
        ("--model-dataset", args.model_dataset),
    ]
    provided = [name for name, value in requested if value]
    if not provided:
        raise RuntimeError(
            "Provide exactly one of: --checkpoint <path> or --model-dataset <dataset_id>."
        )
    if len(provided) > 1:
        raise RuntimeError(
            f"Provide only one checkpoint source, got: {', '.join(provided)}"
        )

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    from clearml import Dataset

    local_dir = Path(
        Dataset.get(dataset_id=args.model_dataset, alias="PlantDiseaseModel").get_local_copy()
    )
    return find_checkpoint_in_dir(local_dir, args.model_filename)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inference and latency benchmark for the PlantVillage ResNet-50 checkpoints."
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a local checkpoint file.")
    parser.add_argument("--model-dataset", default=None,
                        help="ClearML Dataset id that contains the model checkpoint.")
    parser.add_argument("--model-filename", default=None,
                        help="File name of the checkpoint inside --model-dataset "
                             "(default: the only *.pth in the dataset).")
    parser.add_argument("--image", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="./PlantVillage")
    parser.add_argument("--output-dir", default="./results/clean")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument("--queue", default=QUEUE_NAME)
    parser.add_argument("--clearml", action="store_true")
    return parser


def main():
    raw_arguments = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_arguments)

    task = None
    data_root = Path(args.data_dir)

    if args.clearml or running_on_agent():
        task, args, data_root = start_clearml(
            "PlantDisease_Inference", raw_arguments, parser, args
        )

    if not args.image and not args.all:
        parser.error("Choose a mode: --image <path> for one image, or --all for a whole split.")

    device = get_device(args.device)
    checkpoint_path = resolve_checkpoint(args)

    model, bundle, classes = load_bundle(checkpoint_path, device)
    transform, image_size = build_eval_transform(bundle)

    tag = args.tag or str(bundle.get("mode", "model"))
    environment = describe_environment(device)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mode: {bundle.get('mode', 'unknown')} | classes: {len(classes)} | image size: {image_size}")
    print(f"Device: {device} ({environment.get('gpu_name', environment['platform'])})")

    if args.image:
        predictions = predict_single(model, transform, classes, Path(args.image), device, args.topk)
        print(f"\nImage: {args.image}")
        for rank, (name, probability) in enumerate(predictions, start=1):
            print(f"  {rank}. {name:60s} {probability:.4f}")

    if not args.all:
        return

    dataset, dataset_classes, sample_paths = build_split_dataset(
        data_root, args.split, transform, args.limit
    )

    if dataset_classes != classes:
        raise RuntimeError(
            "Class list of the split does not match the checkpoint. "
            f"Split has {len(dataset_classes)} classes, checkpoint has {len(classes)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    targets, predictions, confidences, wall_seconds, peak_memory_mb = run_split(model, loader, device)

    accuracy = float(accuracy_score(targets, predictions))
    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets, predictions, average="weighted", zero_division=0))

    latency = []
    if not args.skip_benchmark:
        pool = build_latency_pool(sample_paths, transform, max(64, args.batch_size))
        for batch_size in (1, args.batch_size):
            iterations = args.benchmark_iterations if batch_size == 1 else max(20, args.benchmark_iterations // 4)
            latency.append(
                measure_latency(model, device, pool, batch_size, args.benchmark_warmup, iterations)
            )

    summary = {
        "tag": tag,
        "mode": bundle.get("mode"),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "samples": int(len(targets)),
        "num_classes": len(classes),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mean_confidence": float(confidences.mean()),
        "mean_confidence_correct": float(confidences[targets == predictions].mean()),
        "mean_confidence_incorrect": float(confidences[targets != predictions].mean())
        if (targets != predictions).any()
        else None,
        "errors": int((targets != predictions).sum()),
        "full_pass_seconds": wall_seconds,
        "peak_gpu_memory_mb": peak_memory_mb,
        "latency": latency,
        "environment": environment,
    }

    output_dir = Path(args.output_dir)
    _, per_class_frame = write_outputs(
        output_dir, tag, sample_paths, classes, targets, predictions, confidences, summary
    )

    print(f"\nSplit: {args.split} | samples: {len(targets)}")
    print(f"Accuracy: {accuracy:.4f} | macro F1: {macro_f1:.4f} | errors: {summary['errors']}")
    print(f"Full pass over the split: {wall_seconds:.1f} s")
    if peak_memory_mb is not None:
        print(f"Peak GPU memory: {peak_memory_mb:.1f} MB")

    for entry in latency:
        print(
            f"Latency batch {entry['batch_size']:>3d}: "
            f"{entry['mean_ms_per_batch']:.2f} ms/batch | "
            f"{entry['mean_ms_per_image']:.2f} ms/image | "
            f"{entry['throughput_images_per_second']:.1f} img/s"
        )

    print("\nFive worst classes by F1:")
    for _, row in per_class_frame.head(5).iterrows():
        print(f"  {row['class']:60s} f1 {row['f1']:.4f} | support {int(row['support'])}")

    print(f"\nOutputs written to {output_dir.resolve()}")

    if task is not None:
        for path in sorted(output_dir.glob(f"*_{tag}.*")):
            task.upload_artifact(
                name=f"inference/{path.name}",
                artifact_object=str(path),
                wait_on_upload=True,
            )


if __name__ == "__main__":
    main()
