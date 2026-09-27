# Plant Disease Recognition from Leaf Images

https://github.com/EgorSavchenko-web/icv2026-plant-disease

Computer Vision 2026, Innopolis University — Project P02.

Fine-tuning versus frozen-backbone linear probing of an ImageNet-pretrained ResNet-50 on
PlantVillage, evaluated on clean laboratory images and on images degraded by simulated robotic field
capture.

## Research question

> Does full fine-tuning of an ImageNet-pretrained ResNet-50 retain its advantage over a frozen
> linear probe when leaf images are degraded by realistic robotic field-capture artifacts?

## Headline results

| Clean test set | Linear probe | Fine-tuned |
|---|---|---|
| Accuracy | 0.9843 | **0.9986** |
| Macro F1 | 0.9804 | **0.9976** |
| Errors out of 2 169 | 34 | **3** |
| Expected calibration error | 0.0080 | **0.0016** |
| Accuracy over three seeds | 0.9822 ± 0.0021 | 0.9988 ± 0.0003 |
| Trainable parameters | 77 862 | 23 585 894 |
| Epochs to early stop | 36 | 16 |

Under degradation the ordering is not constant. Fine-tuning wins under motion blur and JPEG at every
severity, by up to 11 points. Under low light with sensor noise the frozen probe overtakes it
between severity 1 and 2 and is 24 points ahead at severity 2. A control condition — additive
Gaussian noise at unchanged brightness — reproduces the reversal one severity earlier, so noise
alone is sufficient and the darkening and camera gain are not needed. Why noise hurts the
fine-tuned model is a hypothesis the experiments do not test.

Full numbers: `results/main_table.csv`, `results/final_metrics.json`,
`results/robustness/robustness_metrics.csv`, `results/failures/paired_tests.csv`.
Figure: `results/robustness/degradation_accuracy.png`.

## Repository layout

```
1_split_test.py              carve a test split out of train, write its manifest
2_rename_spaces.py           sanitise directory names
3_plant_disease_resnet50.py  training: --mode linear_probe | frozen | finetune
4_logs_building.py           training curves and summary tables from logs
5_inference.py               single-image and full-split inference, latency, per-class metrics
6_robustness_eval.py         evaluation under simulated robotic capture, any number of checkpoints
7_failure_analysis.py        confusion pairs, fragility, Wilson intervals, McNemar tests, ECE
8_collect_results.py         final results tree, main table, run provenance
9_leakage_check.py           perceptual-hash near-duplicate audit of the train/test split
capture_corruptions.py       corruption model shared by 6 and 7

PlantVillage/  train/ val/ test/      dataset (only test/ is committed)
checkpoints/                          model weights, published as release assets
logs/          training/ seeds/ ablation_lr/ ablation_lr_linear_probe/ evaluation/
results/       clean/ comparison/ ablation_lr/ ablation_lr_linear_probe/
               ablation_batchnorm/ robustness/ failures/ leakage/
report/        two-page summary, slides, demo script, project plan
```

## The three training modes

`--mode linear_probe` is the baseline of record: the backbone is frozen **and** every BatchNorm
module is kept in evaluation mode during training, so the features are genuinely fixed.

`--mode frozen` is the same thing without the BatchNorm handling — `requires_grad = False` does not
freeze BatchNorm running statistics, so the backbone still adapts. It is retained deliberately: the
pair isolates the effect of BatchNorm adaptation, and at a matched learning rate it costs 66 errors
against 56.

`--mode finetune` trains all parameters.

Each condition's learning rate was tuned separately on validation macro F1. The linear probe uses
1e-2 (swept 1e-4 to 1e-1, interior optimum); fine-tuning uses 1e-4.

## Environment

Python 3.12, CUDA 12.9. `pip install -r requirements.txt`.

Training and evaluation ran on NVIDIA A100 cards through a ClearML agent using the
`nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04` image, torch 2.10.0+cu129, torchvision 0.25.0+cu129.
`clearml` and `boto3` are needed only for remote execution; every script runs locally without them.

## Dataset

PlantVillage, Kaggle mirror: <https://www.kaggle.com/datasets/mohitsingh1804/plantvillage>

Download and unpack so that `PlantVillage/train` and `PlantVillage/val` exist, then run the two
preprocessing steps **in this order**:

```
python 1_split_test.py        # moves the 2 169 files listed in test_split_manifest.txt into PlantVillage/test
python 2_rename_spaces.py     # replaces spaces in directory and file names with underscores
```

`1_split_test.py` **moves** files rather than copying them: run it exactly once on a fresh
download. The split is defined by the committed `test_split_manifest.txt`, not by a random seed, and
the script moves exactly the files it lists, matching names before or after `2_rename_spaces.py`.
It stops with a list of missing files if any entry is not found.

The split was originally drawn as 5 % of each training class with seed 42, and the manifest was
reconstructed afterwards from the resulting `PlantVillage/test` directory. Seeded sampling depends on
the order in which files are listed, which varies across file systems, so a seed alone would not
reproduce the split reliably; the manifest does. `python 1_split_test.py --random` still draws a
fresh seeded split and writes its own `test_split_manifest.new.txt`.

Resulting split: 41 275 train / 10 861 validation / 2 169 test over 38 classes. `PlantVillage/test`
is committed, so the reported numbers can be verified without downloading the full dataset.

**Known limitation.** The split is not grouped by leaf. PlantVillage contains several photographs of
the same physical leaf, and the dataset's authors kept all images of one leaf on the same side of
their splits; a file-level split does not. Measured with a 64-bit perceptual hash over all 43 444
images: one exact duplicate pair, and 51 test images (2.35 %) within Hamming distance 5 of a
training image, 44 of them in the same class. Removing all 51 moves clean accuracy by at most 0.0001
in either direction, for every run of both conditions; one linear-probe error per seed falls on a
flagged image and no fine-tuned error does. Regenerate the measurement with `9_leakage_check.py`
(writes `results/leakage/`).

## Reproducing the reported results

### Verification path — no training required

Download the checkpoints from release v1.1 into `checkpoints/`:
<https://github.com/EgorSavchenko-web/icv2026-plant-disease/releases/tag/v1.1>

| File | Condition | Seed |
|---|---|---|
| `linear_probe.pth` | linear probe, lr 1e-2 | 42 |
| `linear_probe_s0.pth`, `linear_probe_s1.pth` | linear probe, lr 1e-2 | 0, 1 |
| `finetune.pth` | full fine-tuning, lr 1e-4 | 42 |
| `finetune_s0.pth`, `finetune_s1.pth` | full fine-tuning, lr 1e-4 | 0, 1 |
| `frozen.pth` | BatchNorm ablation: backbone frozen, BatchNorm statistics still updated | 42 |


```
python 5_inference.py --checkpoint checkpoints/linear_probe.pth --all --split test --device cuda
python 5_inference.py --checkpoint checkpoints/finetune.pth     --all --split test --device cuda
python 6_robustness_eval.py --device cuda --checkpoints \
    checkpoints/linear_probe.pth checkpoints/linear_probe_s0.pth checkpoints/linear_probe_s1.pth \
    checkpoints/finetune.pth checkpoints/finetune_s0.pth checkpoints/finetune_s1.pth
python 7_failure_analysis.py --grid-condition low_light:2
python 8_collect_results.py
python 9_leakage_check.py
```

Use `--device cuda` rather than the default `--device auto`: `auto` falls back to CPU when no GPU is
visible, which silently invalidates the latency measurement.

`9_leakage_check.py` needs both `PlantVillage/train` and `PlantVillage/test` and hashes all 43 444
images, which takes roughly half an hour on a CPU. It caches hashes under
`results/leakage/hashes/`, so an interrupted run resumes, and `--budget 120` caps each invocation at
two minutes. Its outputs are committed, so this step is optional for verifying the reported
numbers.

### Single-image prediction, and the demo

The same image can be passed through a capture artifact before prediction. The corruption is seeded
from the image path, so a single-image run reproduces exactly the prediction recorded for that image
in `results/robustness/predictions_robustness.csv`:

```
IMG="PlantVillage/test/Apple___healthy/3aff0cab-c5f0-4fd5-8a7a-2a81160da701___RS_HL_7392.JPG"

python 5_inference.py --checkpoint checkpoints/finetune.pth     --image "$IMG"
python 5_inference.py --checkpoint checkpoints/finetune.pth     --image "$IMG" --corrupt low_light:2
python 5_inference.py --checkpoint checkpoints/linear_probe.pth --image "$IMG" --corrupt low_light:2
```

Clean, the fine-tuned model returns Apple healthy at confidence 1.0000. With the same leaf shot in
shade it returns Tomato Septoria leaf spot, a disease of another crop, at 0.6142. The linear probe,
on that same degraded image, returns Apple healthy at 1.0000.

The image is illustrative, not typical, and was chosen by a stated rule: correct on the clean image
for all six runs, correct under low light severity 2 for all three probe seeds, wrong for all three
fine-tuned seeds with one shared prediction. 202 test images meet that rule. The opposite case, a
fine-tuned model right in every seed where every probe seed is wrong, occurs on 34 images at that
condition, against 331 in this direction. The aggregate numbers above carry the claim; the demo only
shows what one of those failures looks like.

### Full retraining

```
python 3_plant_disease_resnet50.py --mode linear_probe --lr 1e-2 > logs/training/linear_probe.log 2>&1
python 3_plant_disease_resnet50.py --mode finetune     --lr 1e-4 > logs/training/finetune.log 2>&1

python 4_logs_building.py --logs-dir logs/training --out-dir results/comparison
```

Each run writes its checkpoint and artifacts to `outputs/<mode>_<timestamp>/`;
`8_collect_results.py` imports them into `results/` automatically. Approximate cost on one A100:
14.4 min for the linear probe (36 epochs), 9.9 min for fine-tuning (16 epochs). Both use batch 64,
AdamW, `ReduceLROnPlateau` on validation macro F1, early stopping with patience 5, seed 42.

### Remote execution on a ClearML agent

Add `--clearml --queue <queue>` to steps 3, 5 and 6. Checkpoints are fetched with
`--model-dataset <id>` (step 5) or `--from-datasets <id> --model-filenames a.pth b.pth ...` (step 6,
which accepts several checkpoints from one dataset and evaluates them in a single pass).

## Corruption model

Four artifacts of an autonomous field phenotyping robot, applied to the test images in capture
space, before the evaluation transform, with a per-image seed derived from the file path:

| Corruption | Mechanism | Severity 1 / 2 / 3 |
|---|---|---|
| Motion blur | imaging while moving | line kernel 5 / 11 / 19 px, random angle |
| Low light + sensor noise | canopy shade, dusk | exposure 0.50 / 0.30 / 0.18 |
| Gaussian noise (control) | isolates the noise term | σ 0.066 / 0.105 / 0.221 |
| JPEG compression | on-board radio transmission | quality 30 / 15 / 8 |

Low light is not a brightness reduction. The image is darkened, Poisson shot noise and Gaussian read
noise are applied at that reduced signal level, and the result is scaled back up, which emulates a
camera raising sensor gain: mean brightness is restored and the signal-to-noise ratio is destroyed.
Because that mixes three factors, the Gaussian control applies noise alone at unchanged brightness,
with σ matched to the output-referred noise of the low-light condition at mid-grey.

## Results tree

```
results/main_table.csv          headline comparison of all three modes
results/final_metrics.json      every reported number, including the shared inference cost
results/run_provenance.json     ClearML task ids, workers, container, hyperparameters, timings
results/clean/                  per-image predictions, per-class metrics, confusion matrices, curves
results/comparison/             training curves across conditions
results/ablation_lr/            learning-rate sweep, fine-tuned condition
results/ablation_lr_linear_probe/  learning-rate sweep, linear probe
results/ablation_batchnorm/     the naive frozen run, isolating BatchNorm adaptation
results/robustness/             degradation metrics and curves, corruption examples
results/failures/               confusion pairs, fragility, statistics.json, paired_tests.csv
results/leakage/                near-duplicate audit: summary, flagged pairs, nearest neighbours
```

Inference cost is a property of the architecture, so it is reported once. Both conditions of record
measured on the same worker, an A100-PCIE-40 GB: 7.09 and 7.30 ms per image at batch 1, 0.355 ms for
both at batch 64, 773.1 MB peak GPU memory for both.

The `frozen` row of `results/main_table.csv` carries 4.74 ms instead, because that evaluation ran on a
different machine twelve days earlier. Timings across workers are not comparable, so the table keeps
an `inference_worker` column and the reported figure is taken only from the matched pair. The
measurement was left as it was recorded rather than adjusted.

## Team

| Member | Email |
|---|---|
| Egor Savchenko | eg.savchenko@innopolis.university |
| Zamir Safin | z.safin@innopolis.university |
| Ilia Ponomarev | il.ponomarev@innopolis.university |
| Vadim Poponnikov | v.poponnikov@innopolis.university |

Individual contributions are listed in the technical summary in `report/`.
