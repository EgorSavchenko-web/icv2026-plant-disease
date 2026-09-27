import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s*\|\s*"
    r"train loss\s+(?P<train_loss>[0-9.]+)\s*\|\s*"
    r"train accuracy\s+(?P<train_acc>[0-9.]+)\s*\|\s*"
    r"train macro F1\s+(?P<train_f1>[0-9.]+)\s*\|\s*"
    r"val loss\s+(?P<val_loss>[0-9.]+)\s*\|\s*"
    r"val accuracy\s+(?P<val_acc>[0-9.]+)\s*\|\s*"
    r"val macro F1\s+(?P<val_f1>[0-9.]+)\s*\|\s*"
    r"lr\s+(?P<lr>[0-9.eE+-]+)\s*\|\s*"
    r"(?P<time>[0-9.]+)s"
)

FINAL_RE = re.compile(
    r"Final test results\s*\|\s*accuracy\s+(?P<test_acc>[0-9.]+)\s*\|\s*"
    r"macro F1\s+(?P<test_f1>[0-9.]+)\s*\|\s*"
    r"best epoch\s+(?P<best_epoch>\d+)\s*\|\s*"
    r"training time\s+(?P<train_time>[0-9.]+)\s*minutes"
)

MODE_RE = re.compile(r"Training mode:\s*(\w+)")
PARAMS_RE = re.compile(r"Total parameters:\s*(\d+)")
TRAINABLE_RE = re.compile(r"Trainable parameters:\s*(\d+)")
EARLY_RE = re.compile(r"Early stopping at epoch\s+(\d+)")


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#7f7f7f", "#bcbd22",
]


def parse_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    mode_match = MODE_RE.search(text)
    mode = mode_match.group(1) if mode_match else Path(path).stem

    rows = []
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            d = m.groupdict()
            rows.append({
                "epoch": int(d["epoch"]),
                "total_epochs": int(d["total_epochs"]),
                "train_loss": float(d["train_loss"]),
                "train_acc": float(d["train_acc"]),
                "train_f1": float(d["train_f1"]),
                "val_loss": float(d["val_loss"]),
                "val_acc": float(d["val_acc"]),
                "val_f1": float(d["val_f1"]),
                "lr": float(d["lr"]),
                "epoch_time": float(d["time"]),
            })

    if rows:
        df = pd.DataFrame(rows).sort_values("epoch").set_index("epoch")
    else:
        df = pd.DataFrame()

    meta = {
        "mode": mode,
        "file": str(path),
        "total_params": None,
        "trainable_params": None,
        "test_acc": None,
        "test_f1": None,
        "best_epoch": None,
        "train_time_min": None,
        "early_stop_epoch": None,
    }

    m = FINAL_RE.search(text)
    if m:
        meta.update({
            "test_acc": float(m.group("test_acc")),
            "test_f1": float(m.group("test_f1")),
            "best_epoch": int(m.group("best_epoch")),
            "train_time_min": float(m.group("train_time")),
        })

    m = PARAMS_RE.search(text)
    if m:
        meta["total_params"] = int(m.group(1))

    m = TRAINABLE_RE.search(text)
    if m:
        meta["trainable_params"] = int(m.group(1))

    m = EARLY_RE.search(text)
    if m:
        meta["early_stop_epoch"] = int(m.group(1))

    return df, meta


def build_display_names(runs):
    counts = {}
    for _, meta in runs.values():
        base = meta["mode"] or "run"
        counts[base] = counts.get(base, 0) + 1

    names = {}
    for run_id, (_, meta) in runs.items():
        base = meta["mode"] or "run"
        names[run_id] = f"{base} ({run_id})" if counts[base] > 1 else base
    return names


def plot_metric_comparison(runs, display_names, metric, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (run_id, (df, _)) in enumerate(runs.items()):
        if df.empty or metric not in df.columns:
            continue
        ax.plot(
            df.index,
            df[metric],
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            markersize=4,
            linewidth=1.6,
            label=display_names[run_id],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build comparison plots and a summary table from training logs."
    )
    parser.add_argument("--logs-dir", default="logs/training")
    parser.add_argument("--out-dir", default="results/comparison")
    return parser


def main():
    args = build_parser().parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.out_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not logs_dir.exists():
        print(f"[ERROR] Logs directory not found: {logs_dir.resolve()}")
        return

    log_files = sorted(logs_dir.glob("*.log"))
    if not log_files:
        print(f"[WARN] No .log files found in {logs_dir.resolve()}")
        return

    print(f"Found {len(log_files)} log file(s) in {logs_dir.resolve()}")
    print(f"Plots will be saved to: {output_dir.resolve()}")

    runs = {}
    for path in log_files:
        df, meta = parse_log(path)
        run_id = path.stem
        runs[run_id] = (df, meta)

    if not runs:
        print("No data to plot.")
        return

    display_names = build_display_names(runs)

    for run_id, (df, meta) in runs.items():
        print(f"\n=== {display_names[run_id]} ===")
        print(f"File: {meta['file']}")
        print(f"Epochs: {len(df)}")
        if meta["test_acc"] is not None:
            print(
                f"Test accuracy: {meta['test_acc']:.4f} | "
                f"Test macro F1: {meta['test_f1']:.4f} | "
                f"Best epoch: {meta['best_epoch']} | "
                f"Time: {meta['train_time_min']:.1f} min"
            )
        if meta["trainable_params"] is not None:
            print(
                f"Total params: {meta['total_params']:,} | "
                f"Trainable params: {meta['trainable_params']:,}"
            )

    metrics = [
        ("train_loss", "Train loss", "Convergence: train loss"),
        ("val_loss", "Validation loss", "Convergence: validation loss"),
        ("train_acc", "Train accuracy", "Train accuracy"),
        ("val_acc", "Validation accuracy", "Validation accuracy"),
        ("train_f1", "Train macro F1", "Train macro F1"),
        ("val_f1", "Validation macro F1", "Validation macro F1"),
        ("lr", "Learning rate", "Learning rate"),
        ("epoch_time", "Epoch time, sec", "Epoch time"),
    ]

    for metric, ylabel, title in metrics:
        plot_metric_comparison(
            runs,
            display_names,
            metric,
            ylabel,
            title,
            save_path=output_dir / f"compare_{metric}.png",
        )

    run_ids = list(runs.keys())
    labels = [display_names[r] for r in run_ids]
    test_acc = [runs[r][1].get("test_acc") or 0 for r in run_ids]
    test_f1 = [runs[r][1].get("test_f1") or 0 for r in run_ids]

    x = list(range(len(run_ids)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(run_ids) + 3), 5))
    ax.bar([i - width / 2 for i in x], test_acc, width, label="Test accuracy")
    ax.bar([i + width / 2 for i in x], test_f1, width, label="Test macro F1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Value")
    ax.set_title("Final test metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "compare_final_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame({
        display_names[run_id]: runs[run_id][1] for run_id in run_ids
    }).T

    cols = [
        "mode",
        "test_acc",
        "test_f1",
        "best_epoch",
        "train_time_min",
        "total_params",
        "trainable_params",
        "early_stop_epoch",
        "file",
    ]
    summary_table = summary[[c for c in cols if c in summary.columns]]
    print("\n=== Summary ===")
    print(summary_table)

    summary_table.to_csv(output_dir / "summary.csv", encoding="utf-8")
    print(f"\nSummary saved to: {(output_dir / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()