from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights


PROJECT_NAME = os.environ.get("PLANTVILLAGE_CLEARML_PROJECT", "ICV2026_PlantDisease")
QUEUE_NAME = os.environ.get("PLANTVILLAGE_CLEARML_QUEUE", "")
DATASET_ID = os.environ.get("PLANTVILLAGE_DATASET_ID", "687ee9f1c8dd4af98240e55e4b18258a")
TASK_NAME_PREFIX = "PlantDisease_ResNet50"
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger("plant_disease")


logger = setup_logging()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return device


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def load_datasets(data_root: Path):
    train_transform, eval_transform = build_transforms()

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    for path in (train_dir, val_dir, test_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {path}")

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=eval_transform)
    test_dataset = ImageFolder(test_dir, transform=eval_transform)

    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        missing_val = sorted(set(train_dataset.classes) - set(val_dataset.classes))
        missing_test = sorted(set(train_dataset.classes) - set(test_dataset.classes))
        extra_val = sorted(set(val_dataset.classes) - set(train_dataset.classes))
        extra_test = sorted(set(test_dataset.classes) - set(train_dataset.classes))
        raise RuntimeError(
            "Class directories are inconsistent across splits. "
            f"Missing from val: {missing_val}; missing from test: {missing_test}; "
            f"Extra in val: {extra_val}; extra in test: {extra_test}."
        )

    logger.info("Classes: %d", len(train_dataset.classes))
    logger.info("Train samples: %d", len(train_dataset))
    logger.info("Validation samples: %d", len(val_dataset))
    logger.info("Test samples: %d", len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def freeze_backbone_norm(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def build_model(num_classes: int, mode: str):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if mode in ("frozen", "linear_probe"):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif mode == "finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return model


def create_loaders(train_dataset, val_dataset, test_dataset, batch_size: int, num_workers: int, seed: int, device):
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "generator": generator,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool, freeze_bn: bool = False,
              amp_enabled: bool = True):
    model.train(train)

    if train and freeze_bn:
        freeze_backbone_norm(model)

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        predictions = outputs.argmax(dim=1)

        running_loss += loss.item() * inputs.size(0)
        all_targets.append(targets.detach().cpu())
        all_predictions.append(predictions.detach().cpu())

    targets = torch.cat(all_targets).numpy()
    predictions = torch.cat(all_predictions).numpy()
    loss_value = running_loss / len(loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)

    return {
        "loss": float(loss_value),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "targets": targets,
        "predictions": predictions,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_paths = []

    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)

        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = probabilities.argmax(dim=1)

        all_targets.append(targets.cpu())
        all_predictions.append(predictions.cpu())
        all_probabilities.append(probabilities.cpu())

        start = batch_index * loader.batch_size
        stop = start + inputs.size(0)
        all_paths.extend(loader.dataset.samples[start:stop])

    targets = torch.cat(all_targets).numpy()
    predictions = torch.cat(all_predictions).numpy()
    probabilities = torch.cat(all_probabilities).numpy()

    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }

    return metrics, targets, predictions, probabilities, all_paths


def save_confusion_matrix(targets, predictions, class_names, output_path: Path):
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    figure_size = max(10, min(22, len(class_names) * 0.45))

    figure = plt.figure(figsize=(figure_size, figure_size))
    axis = figure.add_subplot(111)
    image = axis.imshow(matrix, interpolation="nearest")
    axis.set_title("Test Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=90, fontsize=7)
    axis.set_yticklabels(class_names, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(history, output_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    figure = plt.figure(figsize=(12, 5))
    axis = figure.add_subplot(111)
    axis.plot(epochs, history["train_loss"], label="Train loss")
    axis.plot(epochs, history["val_loss"], label="Validation loss")
    axis.plot(epochs, history["val_accuracy"], label="Validation accuracy")
    axis.plot(epochs, history["val_macro_f1"], label="Validation macro F1")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Value")
    axis.set_title("Training Curves")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_failure_cases(
    test_dataset,
    targets,
    predictions,
    probabilities,
    output_dir: Path,
    count: int,
):
    failures = np.where(targets != predictions)[0]
    if len(failures) == 0:
        return []

    ranked = sorted(
        failures,
        key=lambda index: probabilities[index, predictions[index]],
        reverse=True,
    )
    selected = ranked[:count]

    failure_dir = output_dir / "failure_cases"
    failure_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for rank, index in enumerate(selected, start=1):
        source_path, _ = test_dataset.samples[index]
        true_name = test_dataset.classes[targets[index]]
        predicted_name = test_dataset.classes[predictions[index]]
        confidence = float(probabilities[index, predictions[index]])

        destination = failure_dir / (
            f"{rank:02d}_true_{true_name}_pred_{predicted_name}_{Path(source_path).name}"
        )
        shutil.copy2(source_path, destination)

        records.append(
            {
                "rank": rank,
                "source_path": str(source_path),
                "saved_path": str(destination),
                "true_class": true_name,
                "predicted_class": predicted_name,
                "prediction_confidence": confidence,
            }
        )

    return records


def report_clearml(task, metrics, mode, epoch=None):
    if task is None:
        return

    logger_reporter = task.get_logger()

    if epoch is None:
        logger_reporter.report_scalar(
            title="Test",
            series=f"{mode} accuracy",
            value=metrics["accuracy"],
            iteration=0,
        )
        logger_reporter.report_scalar(
            title="Test",
            series=f"{mode} macro F1",
            value=metrics["macro_f1"],
            iteration=0,
        )
    else:
        logger_reporter.report_scalar(
            title="Validation",
            series=f"{mode} accuracy",
            value=metrics["accuracy"],
            iteration=epoch,
        )
        logger_reporter.report_scalar(
            title="Validation",
            series=f"{mode} macro F1",
            value=metrics["macro_f1"],
            iteration=epoch,
        )


def upload_outputs(task, output_dir: Path):
    if task is None:
        return

    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            task.upload_artifact(
                name=f"outputs/{path.relative_to(output_dir).as_posix()}",
                artifact_object=str(path),
                wait_on_upload=True,
            )


def running_on_agent():
    return bool(os.environ.get("CLEARML_TASK_ID"))


def transfer_arguments(task, raw_arguments, parser, args):
    payload = {
        "argv": ""
        if running_on_agent()
        else " ".join(__import__("shlex").quote(argument) for argument in raw_arguments)
    }

    task.connect(payload, name="Args")

    if not running_on_agent():
        return args

    if not payload["argv"]:
        raise RuntimeError("No command line arguments were stored in the ClearML task.")

    return parser.parse_args(__import__("shlex").split(payload["argv"]))


def start_clearml(task_name, raw_arguments, parser, args):
    from clearml import Dataset, Task

    if DATASET_ID == "PASTE_DATASET_ID_HERE":
        raise RuntimeError("Set PLANTVILLAGE_DATASET_ID to the ClearML dataset ID before running remotely.")

    task = Task.init(project_name=PROJECT_NAME, task_name=task_name)
    args = transfer_arguments(task, raw_arguments, parser, args)

    if not running_on_agent():
        os.environ["CLEARML_APT_INSTALL"] = ""

        task.set_packages(
            [
                "clearml>=2.1",
                "boto3",
                "numpy>=1.26",
                "scikit-learn>=1.4",
                "matplotlib>=3.8",
                "torch==2.10.0",
                "torchvision==0.25.0",
            ]
        )

        task.execute_remotely(queue_name=args.queue, exit_process=True)

    data_dir = Path(Dataset.get(dataset_id=DATASET_ID, alias="PlantVillage").get_local_copy())
    logger.info("Dataset materialised at %s", data_dir)

    return task, args, data_dir


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plant disease recognition with pretrained ResNet-50."
    )
    parser.add_argument("--mode", choices=["frozen", "linear_probe", "finetune"], default="frozen",
                        help="frozen: backbone weights fixed but its BatchNorm running statistics "
                             "still adapt. linear_probe: backbone fully fixed, BatchNorm kept in "
                             "eval mode, i.e. a true linear probe on frozen features. finetune: "
                             "all parameters trainable.")
    parser.add_argument("--data-dir", default="./PlantVillage")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queue", default=QUEUE_NAME)
    parser.add_argument("--clearml", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main():
    raw_arguments = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_arguments)

    set_seed(args.seed)

    task = None
    data_root = Path(args.data_dir)

    if args.clearml or running_on_agent():
        task_name = f"{TASK_NAME_PREFIX}_{args.mode}"
        task, args, data_root = start_clearml(task_name, raw_arguments, parser, args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_output = Path("./outputs") / f"{args.mode}_{timestamp}"
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    logger.info("Using device: %s", device)
    logger.info("Training mode: %s", args.mode)

    train_dataset, val_dataset, test_dataset = load_datasets(data_root)

    train_loader, val_loader, test_loader = create_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        args.batch_size,
        args.num_workers,
        args.seed,
        device,
    )

    model = build_model(len(train_dataset.classes), args.mode).to(device)

    trainable_parameters = count_trainable_parameters(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    logger.info("Total parameters: %d", total_parameters)
    logger.info("Trainable parameters: %d", trainable_parameters)

    if task is not None:
        task.connect(
            {
                "mode": args.mode,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "image_size": IMAGE_SIZE,
                "seed": args.seed,
                "num_classes": len(train_dataset.classes),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
            },
            name="Configuration",
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(device.type, enabled=amp_enabled)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }

    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_result = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            train=True,
            freeze_bn=args.mode == "linear_probe",
            amp_enabled=amp_enabled,
        )

        val_result = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            scaler,
            train=False,
            amp_enabled=amp_enabled,
        )

        scheduler.step(val_result["macro_f1"])

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_result["loss"])
        history["train_accuracy"].append(train_result["accuracy"])
        history["train_macro_f1"].append(train_result["macro_f1"])
        history["val_loss"].append(val_result["loss"])
        history["val_accuracy"].append(val_result["accuracy"])
        history["val_macro_f1"].append(val_result["macro_f1"])
        history["learning_rate"].append(current_lr)

        report_clearml(task, val_result, args.mode, epoch)

        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch %d/%d | train loss %.4f | train accuracy %.4f | train macro F1 %.4f | "
            "val loss %.4f | val accuracy %.4f | val macro F1 %.4f | lr %.6g | %.1fs",
            epoch,
            args.epochs,
            train_result["loss"],
            train_result["accuracy"],
            train_result["macro_f1"],
            val_result["loss"],
            val_result["accuracy"],
            val_result["macro_f1"],
            current_lr,
            elapsed,
        )

        if val_result["macro_f1"] > best_score:
            best_score = val_result["macro_f1"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            logger.info("Early stopping at epoch %d.", epoch)
            break

    training_time = time.time() - training_start

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)

    torch.save(
        {
            "bundle_version": 1,
            "kind": "resnet50_plant_disease",
            "mode": args.mode,
            "image_size": IMAGE_SIZE,
            "classes": list(train_dataset.classes),
            "class_to_idx": dict(train_dataset.class_to_idx),
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
            "model_state_dict": best_state,
            "num_classes": len(train_dataset.classes),
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "best_epoch": best_epoch,
            "configuration": {
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
            },
        },
        output_dir / "best_model.pth",
    )

    test_metrics, test_targets, test_predictions, test_probabilities, test_paths = evaluate(
        model,
        test_loader,
        device,
    )

    report = classification_report(
        test_targets,
        test_predictions,
        target_names=train_dataset.classes,
        zero_division=0,
        output_dict=True,
    )

    confusion_path = output_dir / "confusion_matrix.png"
    curves_path = output_dir / "training_curves.png"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "classification_report.json"

    save_confusion_matrix(
        test_targets,
        test_predictions,
        train_dataset.classes,
        confusion_path,
    )
    save_training_curves(history, curves_path)

    failure_cases = save_failure_cases(
        test_dataset,
        test_targets,
        test_predictions,
        test_probabilities,
        output_dir,
        count=3,
    )

    results = {
        "mode": args.mode,
        "dataset": "PlantVillage",
        "best_epoch": best_epoch,
        "training_time_seconds": training_time,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "num_classes": len(train_dataset.classes),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "classes": train_dataset.classes,
        "failure_cases": failure_cases,
        "configuration": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "image_size": IMAGE_SIZE,
            "seed": args.seed,
        },
        "history": history,
    }

    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report_clearml(task, test_metrics, args.mode)

    if task is not None:
        upload_outputs(task, output_dir)

    logger.info(
        "Final test results | accuracy %.4f | macro F1 %.4f | best epoch %d | training time %.1f minutes",
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
        best_epoch,
        training_time / 60.0,
    )
    logger.info("Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
