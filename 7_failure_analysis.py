from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import binomtest

from capture_corruptions import condition_label, corrupt_image


MODEL_COLORS = {"frozen": "#0072B2", "finetune": "#D55E00"}
MODEL_LABELS = {"frozen": "Frozen backbone", "finetune": "Fine-tuned"}
FALLBACK_COLORS = ["#009E73", "#CC79A7", "#56B4E9"]


def split_class_name(name: str):
    if "___" in name:
        crop, disease = name.split("___", 1)
        return crop, disease
    return name, name


def shorten(name: str) -> str:
    crop, disease = split_class_name(name)
    return f"{crop.replace('_', ' ')}: {disease.replace('_', ' ')}"


def wilson_interval(successes: int, total: int, z: float = 1.96):
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def expected_calibration_error(confidence, correct, bins: int = 15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def paired_comparison(frame: pd.DataFrame, first: str, second: str, corruption: str, severity: int):
    a = frame[(frame["model"] == first) & (frame["corruption"] == corruption)
              & (frame["severity"] == severity)].sort_values("path")
    b = frame[(frame["model"] == second) & (frame["corruption"] == corruption)
              & (frame["severity"] == severity)].sort_values("path")

    if len(a) == 0 or len(a) != len(b):
        return None

    first_only = int((a["correct"].values & ~b["correct"].values).sum())
    second_only = int((~a["correct"].values & b["correct"].values).sum())
    discordant = first_only + second_only

    return {
        "corruption": corruption,
        "severity": severity,
        f"{first}_only_correct": first_only,
        f"{second}_only_correct": second_only,
        "discordant": discordant,
        "winner": second if second_only > first_only else first if first_only > second_only else "tie",
        "p_value": float(binomtest(second_only, discordant, 0.5).pvalue) if discordant else 1.0,
    }


def build_statistics(frame: pd.DataFrame, models):
    statistics = {"per_condition": [], "paired": []}

    for model in models:
        for (corruption, severity), group in frame[frame["model"] == model].groupby(
                ["corruption", "severity"]):
            correct = group["correct"].values.astype(bool)
            confidence = group["confidence"].values
            successes, total = int(correct.sum()), len(group)
            low, high = wilson_interval(successes, total)
            wrong = confidence[~correct]

            statistics["per_condition"].append({
                "model": model,
                "corruption": corruption,
                "severity": int(severity),
                "n": total,
                "accuracy": successes / total,
                "accuracy_ci95_low": low,
                "accuracy_ci95_high": high,
                "errors": total - successes,
                "ece": expected_calibration_error(confidence, correct.astype(float)),
                "mean_confidence": float(confidence.mean()),
                "mean_confidence_when_wrong": float(wrong.mean()) if len(wrong) else None,
                "n_errors_behind_that_mean": int(len(wrong)),
            })

    if len(models) == 2:
        first, second = sorted(models)
        for (corruption, severity) in sorted(
                {(c, int(s)) for c, s in zip(frame["corruption"], frame["severity"])}):
            result = paired_comparison(frame, first, second, corruption, severity)
            if result:
                statistics["paired"].append(result)

    return statistics


def resolve_local_path(stored_path: str, data_dir: Path) -> Path:
    parts = Path(stored_path).parts[-3:]
    return data_dir.joinpath(*parts)


def corruption_key(stored_path: str) -> str:
    parts = Path(stored_path).parts[-2:]
    return "/".join(parts)


def confusion_pairs(frame: pd.DataFrame, top: int) -> pd.DataFrame:
    errors = frame[~frame["correct"]]

    grouped = (
        errors.groupby(["model", "corruption", "severity", "true_class", "predicted_class"])
        .size()
        .reset_index(name="count")
    )

    grouped["same_crop"] = grouped.apply(
        lambda row: split_class_name(row["true_class"])[0]
        == split_class_name(row["predicted_class"])[0],
        axis=1,
    )

    grouped = grouped.sort_values(
        ["model", "corruption", "severity", "count"], ascending=[True, True, True, False]
    )

    return grouped.groupby(["model", "corruption", "severity"], group_keys=False).head(top)


def class_f1_drop(per_class: pd.DataFrame) -> pd.DataFrame:
    clean = per_class[per_class["corruption"] == "none"][["model", "class", "f1"]]
    clean = clean.rename(columns={"f1": "clean_f1"})

    corrupted = per_class[per_class["corruption"] != "none"]
    merged = corrupted.merge(clean, on=["model", "class"], how="left")
    merged["f1_drop"] = merged["clean_f1"] - merged["f1"]

    return merged.sort_values("f1_drop", ascending=False)


def render_failure_grid(frame: pd.DataFrame, models, columns: int, corruption: str, severity: int, data_dir: Path, output_path: Path):
    plt.rcParams.update({"font.size": 6})

    figure, axes = plt.subplots(
        len(models), columns, figsize=(1.5 * columns, 2.45 * len(models)), squeeze=False
    )

    for row_index, model in enumerate(models):
        subset = frame[(frame["model"] == model) & (~frame["correct"])]
        subset = subset.sort_values("confidence", ascending=False).head(columns)

        for column_index in range(columns):
            axis = axes[row_index][column_index]
            axis.set_xticks([])
            axis.set_yticks([])

            if column_index >= len(subset):
                axis.axis("off")
                continue

            row = subset.iloc[column_index]
            local_path = resolve_local_path(row["path"], data_dir)
            image = Image.open(local_path).convert("RGB")
            image = corrupt_image(image, corruption, severity, corruption_key(row["path"]))

            axis.imshow(image)

            caption_lines = [
                f"true: {shorten(row['true_class'])}",
                f"pred: {shorten(row['predicted_class'])}",
                f"p = {row['confidence']:.3f}",
            ]
            wrapped = []
            for line in caption_lines:
                wrapped.extend(textwrap.wrap(line, width=30) or [line])

            axis.set_title("\n".join(wrapped), fontsize=5, loc="left")

        axes[row_index][0].set_ylabel(MODEL_LABELS.get(model, model), fontsize=7)

    figure.suptitle(f"Most confident errors - {condition_label(corruption, severity)}", fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.subplots_adjust(hspace=0.55)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_fragile_classes(drops: pd.DataFrame, models, corruption: str, severity: int, top: int, output_path: Path):
    plt.rcParams.update({"font.size": 7})

    subset = drops[(drops["corruption"] == corruption) & (drops["severity"] == severity)]
    reference = subset[subset["model"] == models[0]].sort_values("f1_drop", ascending=False).head(top)
    order = reference["class"].tolist()[::-1]

    figure, axis = plt.subplots(figsize=(7.0, 0.28 * len(order) + 1.2))
    positions = range(len(order))
    height = 0.38

    for model_index, model in enumerate(models):
        values = [
            float(subset[(subset["model"] == model) & (subset["class"] == name)]["f1_drop"].iloc[0])
            if not subset[(subset["model"] == model) & (subset["class"] == name)].empty
            else 0.0
            for name in order
        ]
        offset = (model_index - (len(models) - 1) / 2) * height
        axis.barh(
            [position + offset for position in positions],
            values,
            height=height * 0.92,
            color=MODEL_COLORS.get(model, FALLBACK_COLORS[model_index % len(FALLBACK_COLORS)]),
            label=MODEL_LABELS.get(model, model),
        )

    axis.set_yticks(list(positions))
    axis.set_yticklabels([shorten(name) for name in order], fontsize=6)
    axis.set_xlabel(f"Per-class F1 drop from clean - {condition_label(corruption, severity)}")
    axis.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=len(models))

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_condition(value: str):
    if ":" not in value:
        raise argparse.ArgumentTypeError("Use the form corruption:severity, for example motion_blur:2")
    corruption, severity = value.split(":", 1)
    return corruption, int(severity)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Failure and confusion analysis for the PlantVillage ResNet-50 comparison."
    )
    parser.add_argument("--predictions", default="./results/robustness/predictions_robustness.csv")
    parser.add_argument("--per-class", default="./results/robustness/per_class_under_corruption.csv")
    parser.add_argument("--output-dir", default="./results/failures")
    parser.add_argument("--data-dir", default="./PlantVillage")
    parser.add_argument("--top-pairs", type=int, default=10)
    parser.add_argument("--grid-columns", type=int, default=5)
    parser.add_argument("--grid-condition", type=parse_condition, default=("motion_blur", 2))
    parser.add_argument("--fragile-top", type=int, default=12)
    return parser


def main():
    args = build_parser().parse_args()

    predictions = pd.read_csv(args.predictions)
    per_class = pd.read_csv(args.per_class)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(predictions["model"].unique())

    pairs = confusion_pairs(predictions, args.top_pairs)
    pairs.to_csv(output_dir / "top_confusion_pairs.csv", index=False, encoding="utf-8")

    drops = class_f1_drop(per_class)
    drops.to_csv(output_dir / "class_f1_drop.csv", index=False, encoding="utf-8")

    data_dir = Path(args.data_dir)

    clean_predictions = predictions[predictions["corruption"] == "none"]
    render_failure_grid(
        clean_predictions,
        models,
        args.grid_columns,
        "none",
        0,
        data_dir,
        output_dir / "failure_grid_clean.png",
    )

    corruption, severity = args.grid_condition
    corrupted_predictions = predictions[
        (predictions["corruption"] == corruption) & (predictions["severity"] == severity)
    ]
    render_failure_grid(
        corrupted_predictions,
        models,
        args.grid_columns,
        corruption,
        severity,
        data_dir,
        output_dir / f"failure_grid_{corruption}_s{severity}.png",
    )

    plot_fragile_classes(
        drops, models, corruption, severity, args.fragile_top, output_dir / "fragile_classes.png"
    )

    statistics = build_statistics(predictions, models)
    (output_dir / "statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    pd.DataFrame(statistics["per_condition"]).to_csv(
        output_dir / "per_condition_statistics.csv", index=False, encoding="utf-8")
    pd.DataFrame(statistics["paired"]).to_csv(
        output_dir / "paired_tests.csv", index=False, encoding="utf-8")

    summary = {"models": models, "grid_condition": condition_label(corruption, severity)}

    for model in models:
        clean_model = clean_predictions[clean_predictions["model"] == model]
        clean_errors = clean_model[~clean_model["correct"]]
        model_pairs = pairs[(pairs["model"] == model) & (pairs["corruption"] == "none")]

        summary[model] = {
            "clean_errors": int(len(clean_errors)),
            "clean_error_rate": float(len(clean_errors) / max(len(clean_model), 1)),
            "same_crop_error_share": float(model_pairs["same_crop"].mul(model_pairs["count"]).sum()
                                           / max(model_pairs["count"].sum(), 1)),
            "top_confusions": [
                {
                    "true_class": row["true_class"],
                    "predicted_class": row["predicted_class"],
                    "count": int(row["count"]),
                    "same_crop": bool(row["same_crop"]),
                }
                for row in model_pairs.head(5).to_dict("records")
            ],
        }

    (output_dir / "failure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Clean confusion pairs ===")
    for model in models:
        print(f"\n{MODEL_LABELS.get(model, model)}")
        model_pairs = pairs[(pairs["model"] == model) & (pairs["corruption"] == "none")]
        for row in model_pairs.head(args.top_pairs).to_dict("records"):
            marker = "same crop" if row["same_crop"] else "cross crop"
            print(
                f"  {row['count']:3d}x {shorten(row['true_class'])} -> "
                f"{shorten(row['predicted_class'])}  [{marker}]"
            )

    print("\n=== Paired tests, identical inputs for both models ===")
    for row in statistics["paired"]:
        print(f"  {row['corruption']:14s} s{row['severity']}: winner {row['winner']:9s} "
              f"discordant {row['discordant']:4d}  p = {row['p_value']:.3g}")

    print("\n=== Calibration (expected calibration error) ===")
    for row in statistics["per_condition"]:
        if row["corruption"] in ("none", "low_light") and row["severity"] in (0, 2):
            print(f"  {row['model']:9s} {row['corruption']:11s} s{row['severity']}: "
                  f"ECE {row['ece']:.4f} | acc {row['accuracy']:.4f} "
                  f"[{row['accuracy_ci95_low']:.4f}, {row['accuracy_ci95_high']:.4f}]")

    print(f"\nOutputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
