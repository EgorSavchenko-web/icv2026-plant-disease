from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


from capture_corruptions import (
    CORRUPTION_LABELS,
    CORRUPTION_PARAMETERS,
    CORRUPTIONS,
    SEVERITIES,
    all_conditions,
    condition_label,
    corrupt_image,
)


PROJECT_NAME = os.environ.get("PLANTVILLAGE_CLEARML_PROJECT", "ICV2026_PlantDisease")
QUEUE_NAME = os.environ.get("PLANTVILLAGE_CLEARML_QUEUE", "")
DATASET_ID = os.environ.get("PLANTVILLAGE_DATASET_ID", "687ee9f1c8dd4af98240e55e4b18258a")
FALLBACK_MEAN = [0.485, 0.456, 0.406]
FALLBACK_STD = [0.229, 0.224, 0.225]

MODEL_COLORS = {"frozen": "#0072B2", "linear_probe": "#0072B2", "finetune": "#D55E00"}
MODEL_MARKERS = {"frozen": "o", "linear_probe": "o", "finetune": "s"}
MODEL_LINESTYLES = {"frozen": "-", "linear_probe": "-", "finetune": "--"}
FALLBACK_COLORS = ["#009E73", "#CC79A7", "#56B4E9"]
MODEL_LABELS = {"frozen": "Frozen backbone", "linear_probe": "Linear probe", "finetune": "Fine-tuned"}


def base_condition(name: str) -> str:
    return name.rsplit("_s", 1)[0] if "_s" in name and name.rsplit("_s", 1)[1].isdigit() else name


class CorruptedImageFolder(ImageFolder):
    def __init__(self, root, transform, corruption: str, severity: int):
        super().__init__(root, transform=transform)
        self.corruption = corruption
        self.severity = severity
        self.root_path = Path(root)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        key = Path(path).relative_to(self.root_path).as_posix()
        image = corrupt_image(image, self.corruption, self.severity, key)
        return self.transform(image), target


def get_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return device


def load_model(checkpoint_path: Path, device):
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def evaluate_condition(entries, loader, device):
    collected = {name: {"predictions": [], "confidences": []} for name, _ in entries}
    targets = []

    with torch.inference_mode():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets.append(batch_targets)

            for name, model in entries:
                probabilities = torch.softmax(model(inputs), dim=1)
                confidences, predictions = probabilities.max(dim=1)
                collected[name]["predictions"].append(predictions.cpu())
                collected[name]["confidences"].append(confidences.cpu())

    targets = torch.cat(targets).numpy()

    for name in collected:
        collected[name]["predictions"] = torch.cat(collected[name]["predictions"]).numpy()
        collected[name]["confidences"] = torch.cat(collected[name]["confidences"]).numpy()

    return targets, collected


def per_class_f1(targets, predictions, classes):
    scores = f1_score(
        targets,
        predictions,
        labels=list(range(len(classes))),
        average=None,
        zero_division=0,
    )
    return {classes[index]: float(score) for index, score in enumerate(scores)}


def model_style(name: str, index: int):
    color = MODEL_COLORS.get(name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])
    marker = MODEL_MARKERS.get(name, "^")
    linestyle = MODEL_LINESTYLES.get(name, ":")
    return color, marker, linestyle


def plot_degradation(metrics_frame: pd.DataFrame, model_names, output_path: Path, metric_rows=None):
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})

    metric_rows = metric_rows or [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]
    height = 2.35 if len(metric_rows) == 1 else 4.2

    figure, axes = plt.subplots(
        len(metric_rows), len(CORRUPTIONS), figsize=(7.0, height),
        sharex=True, sharey="row", squeeze=False,
    )

    for row_index, (metric, metric_label) in enumerate(metric_rows):
        for column_index, corruption in enumerate(CORRUPTIONS):
            axis = axes[row_index][column_index]

            conditions = []
            for name in model_names:
                base = base_condition(name)
                if base not in conditions:
                    conditions.append(base)

            for condition_index, base in enumerate(conditions):
                color, marker, linestyle = model_style(base, condition_index)
                members = [n for n in model_names if base_condition(n) == base]

                series = []
                for name in members:
                    clean_value = metrics_frame[
                        (metrics_frame["model"] == name) & (metrics_frame["corruption"] == "none")
                    ][metric].iloc[0]
                    subset = metrics_frame[
                        (metrics_frame["model"] == name)
                        & (metrics_frame["corruption"] == corruption)
                    ].sort_values("severity")
                    series.append([clean_value] + subset[metric].tolist())

                values = np.asarray(series, dtype=float)
                x_values = list(range(values.shape[1]))
                mean = values.mean(axis=0)

                if values.shape[0] > 1:
                    axis.fill_between(x_values, values.min(axis=0), values.max(axis=0),
                                      color=color, alpha=0.18, linewidth=0)

                label = MODEL_LABELS.get(base, base)
                if values.shape[0] > 1:
                    label = f"{label} (n={values.shape[0]})"

                axis.plot(
                    x_values,
                    mean,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.4,
                    markersize=4.0,
                    label=label,
                )

            axis.set_xticks([0, 1, 2, 3])
            axis.grid(True, alpha=0.25, linewidth=0.5)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

            if row_index == 0:
                axis.set_title(CORRUPTION_LABELS[corruption], fontsize=8)
            if row_index == len(metric_rows) - 1:
                axis.set_xlabel("Severity (0 = clean)")
            if column_index == 0:
                axis.set_ylabel(metric_label)

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    figure.tight_layout(rect=(0, 0.11 if len(metric_rows) == 1 else 0.06, 1, 1))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_corruption_examples(image_path: Path, output_path: Path):
    plt.rcParams.update({"font.size": 7})

    source = Image.open(image_path).convert("RGB")
    key = image_path.name

    figure, axes = plt.subplots(len(CORRUPTIONS), len(SEVERITIES) + 1, figsize=(6.0, 4.6))

    for row_index, corruption in enumerate(CORRUPTIONS):
        for column_index in range(len(SEVERITIES) + 1):
            axis = axes[row_index][column_index]

            if column_index == 0:
                axis.imshow(source)
                title = "Clean"
            else:
                severity = SEVERITIES[column_index - 1]
                axis.imshow(corrupt_image(source, corruption, severity, key))
                title = f"s{severity}"

            axis.set_xticks([])
            axis.set_yticks([])

            if row_index == 0:
                axis.set_title(title, fontsize=7)
            if column_index == 0:
                axis.set_ylabel(CORRUPTION_LABELS[corruption], fontsize=7)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def render_figures(metrics_frame: pd.DataFrame, model_names, output_dir: Path, args, example_path=None):
    plot_degradation(metrics_frame, model_names, output_dir / "degradation_curves.png")
    plot_degradation(
        metrics_frame,
        model_names,
        output_dir / "degradation_accuracy.png",
        metric_rows=[("accuracy", "Accuracy")],
    )

    example = Path(args.example_image) if args.example_image else example_path
    if example is not None:
        plot_corruption_examples(example, output_dir / "corruption_examples.png")


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


def start_clearml(raw_arguments, parser, args):
    from clearml import Dataset, Task

    task = Task.init(project_name=PROJECT_NAME, task_name="PlantDisease_Robustness")
    args = transfer_arguments(task, raw_arguments, parser, args)

    if not running_on_agent():
        os.environ["CLEARML_APT_INSTALL"] = ""
        task.set_packages(
            [
                "clearml>=2.1",
                "boto3",
                "numpy>=1.26",
                "pandas>=2.0",
                "scipy>=1.11",
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
    """Locate the checkpoint file inside a ClearML dataset directory."""
    if filename:
        candidate = local_dir / filename
        if candidate.is_file():
            return candidate
        matches = [p for p in local_dir.rglob(filename) if p.is_file()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"File '{filename}' not found in dataset at {local_dir}")
        raise RuntimeError(f"Multiple files named '{filename}' found in dataset: {matches}")

    candidates = sorted(p for p in local_dir.rglob("*.pth") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No .pth files found in dataset at {local_dir}")
    if len(candidates) > 1:
        names = ", ".join(p.relative_to(local_dir).as_posix() for p in candidates)
        raise RuntimeError(
            f"Multiple .pth files found in dataset: {names}. "
            f"Pass --model-filenames to disambiguate."
        )
    return candidates[0]


def resolve_checkpoints(args):
    if args.from_datasets:
        from clearml import Dataset

        filenames = args.model_filenames or [None] * len(args.from_datasets)
        dataset_ids = list(args.from_datasets)

        if len(dataset_ids) == 1 and len(filenames) > 1:
            dataset_ids = dataset_ids * len(filenames)

        if len(filenames) != len(dataset_ids):
            raise RuntimeError(
                "--model-filenames must have the same length as --from-datasets, or a single "
                f"dataset must be given with several file names ({len(filenames)} vs "
                f"{len(dataset_ids)})."
            )

        resolved = []
        for dataset_id, filename in zip(dataset_ids, filenames):
            local_dir = Path(
                Dataset.get(dataset_id=dataset_id, alias="PlantDiseaseModel").get_local_copy()
            )
            resolved.append(find_checkpoint_in_dir(local_dir, filename))
        return resolved

    if not args.checkpoints:
        raise RuntimeError("Provide --from-datasets or --checkpoints.")

    resolved = []
    for entry in args.checkpoints:
        path = Path(entry)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        resolved.append(path)

    return resolved


def build_parser():
    parser = argparse.ArgumentParser(
        description="Robustness of the PlantVillage ResNet-50 checkpoints under simulated robotic capture artifacts."
    )
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Local paths to checkpoint files.")
    parser.add_argument("--from-datasets", nargs="+", default=None,
                        help="ClearML Dataset ids containing the model checkpoints.")
    parser.add_argument("--model-filenames", nargs="+", default=None,
                        help="Optional per-dataset file names; if omitted, each dataset is "
                             "searched for its single *.pth file.")
    parser.add_argument("--data-dir", default="./PlantVillage")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="./results/robustness")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--example-image", default=None)
    parser.add_argument("--plots-only", action="store_true",
                        help="Redraw the figures from an existing robustness_metrics.csv "
                             "without running inference.")
    parser.add_argument("--queue", default=QUEUE_NAME)
    parser.add_argument("--clearml", action="store_true")
    return parser


def main():
    raw_arguments = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_arguments)

    if args.plots_only:
        output_dir = Path(args.output_dir)
        metrics_frame = pd.read_csv(output_dir / "robustness_metrics.csv")
        model_names = sorted(metrics_frame["model"].unique())
        render_figures(metrics_frame, model_names, output_dir, args)
        print(f"Figures rewritten in {output_dir.resolve()}")
        return

    task = None
    data_root = Path(args.data_dir)

    if args.clearml or running_on_agent():
        task, args, data_root = start_clearml(raw_arguments, parser, args)

    device = get_device(args.device)
    checkpoint_paths = resolve_checkpoints(args)

    print(f"Device: {device}"
          f"{' (' + torch.cuda.get_device_name(device) + ')' if device.type == 'cuda' else ''}")

    entries = []
    reference_classes = None
    transform = None

    loaded = [(path,) + load_model(path, device) for path in checkpoint_paths]
    mode_counts = {}
    for _, _, bundle, _ in loaded:
        mode = str(bundle.get("mode", "model"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    for path, model, bundle, classes in loaded:
        mode = str(bundle.get("mode", path.stem))
        seed = bundle.get("configuration", {}).get("seed")
        name = mode if mode_counts[mode] == 1 else f"{mode}_s{seed}"

        if reference_classes is None:
            reference_classes = classes
            transform = build_eval_transform(bundle)
        elif classes != reference_classes:
            raise RuntimeError(f"Checkpoint {path} has a different class list than the first one.")

        entries.append((name, model))
        print(f"Loaded {name} from {path}")

    model_names = [name for name, _ in entries]
    split_dir = data_root / args.split

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    metric_rows = []
    per_class_rows = []
    prediction_frames = []

    for corruption, severity in all_conditions():
        dataset = CorruptedImageFolder(split_dir, transform, corruption, severity)

        if dataset.classes != reference_classes:
            raise RuntimeError("Class list of the split does not match the checkpoints.")

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        started = time.perf_counter()
        targets, collected = evaluate_condition(entries, loader, device)
        elapsed = time.perf_counter() - started

        sample_paths = [Path(path).as_posix() for path, _ in dataset.samples]

        for name in model_names:
            predictions = collected[name]["predictions"]
            confidences = collected[name]["confidences"]

            metric_rows.append(
                {
                    "model": name,
                    "corruption": corruption,
                    "severity": severity,
                    "condition": condition_label(corruption, severity),
                    "accuracy": float(accuracy_score(targets, predictions)),
                    "macro_f1": float(
                        f1_score(targets, predictions, average="macro", zero_division=0)
                    ),
                    "mean_confidence": float(confidences.mean()),
                    "errors": int((targets != predictions).sum()),
                    "samples": int(len(targets)),
                }
            )

            for class_name, score in per_class_f1(targets, predictions, reference_classes).items():
                per_class_rows.append(
                    {
                        "model": name,
                        "corruption": corruption,
                        "severity": severity,
                        "class": class_name,
                        "f1": score,
                    }
                )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "corruption": corruption,
                        "severity": severity,
                        "path": sample_paths,
                        "true_class": [reference_classes[int(index)] for index in targets],
                        "predicted_class": [reference_classes[int(index)] for index in predictions],
                        "confidence": confidences,
                        "correct": targets == predictions,
                    }
                )
            )

        print(f"{condition_label(corruption, severity):32s} done in {elapsed:.1f} s")

    metrics_frame = pd.DataFrame(metric_rows)

    clean_lookup = {
        row["model"]: row
        for row in metrics_frame[metrics_frame["corruption"] == "none"].to_dict("records")
    }

    metrics_frame["accuracy_drop"] = metrics_frame.apply(
        lambda row: clean_lookup[row["model"]]["accuracy"] - row["accuracy"], axis=1
    )
    metrics_frame["relative_accuracy_drop"] = metrics_frame.apply(
        lambda row: row["accuracy_drop"] / clean_lookup[row["model"]]["accuracy"], axis=1
    )
    metrics_frame["macro_f1_drop"] = metrics_frame.apply(
        lambda row: clean_lookup[row["model"]]["macro_f1"] - row["macro_f1"], axis=1
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_frame.to_csv(output_dir / "robustness_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(per_class_rows).to_csv(
        output_dir / "per_class_under_corruption.csv", index=False, encoding="utf-8"
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "predictions_robustness.csv", index=False, encoding="utf-8"
    )

    render_figures(
        metrics_frame,
        model_names,
        output_dir,
        args,
        example_path=Path(dataset.samples[0][0]),
    )

    (output_dir / "corruption_settings.json").write_text(
        json.dumps(CORRUPTION_PARAMETERS, indent=2), encoding="utf-8"
    )

    print("\n=== Robustness summary ===")
    summary_view = metrics_frame.pivot_table(
        index=["corruption", "severity"], columns="model", values="accuracy"
    )
    print(summary_view.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nOutputs written to {output_dir.resolve()}")

    if task is not None:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                task.upload_artifact(
                    name=f"robustness/{path.name}", artifact_object=str(path), wait_on_upload=True
                )


if __name__ == "__main__":
    main()
