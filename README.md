# Plant Disease Recognition from Leaf Images

https://github.com/EgorSavchenko-web/icv2026-plant-disease

Computer Vision 2026, Innopolis University — Project P02.

Fine-tuning versus frozen-backbone transfer of an ImageNet-pretrained ResNet-50 on PlantVillage,
evaluated on clean laboratory images and on images degraded by simulated robotic field capture.

## Research question

> Does full fine-tuning of an ImageNet-pretrained ResNet-50 retain its advantage over
> frozen-backbone linear probing when leaf images are degraded by realistic robotic field-capture
> artifacts rather than captured under laboratory conditions?

Sub-questions: the clean-data gap between the two conditions; sensitivity to the learning rate;
the degradation profile under three independent capture artifacts at three severities; and which
visually similar disease pairs are confused first.

## Headline results

| | Frozen backbone | Fine-tuned |
|---|---|---|
| Test accuracy | 0.9696 | **0.9986** |
| Test macro F1 | 0.9664 | **0.9976** |
| Errors out of 2 169 | 66 | 3 |
| Trainable parameters | 77 862 | 23 585 894 |
| Epochs to early stop | 47 | 16 |
| Training time | 14.7 min | 9.9 min |
| Mean confidence when wrong | 0.531 | 0.859 |

Under degradation the ordering is not constant. Fine-tuning wins under motion blur and JPEG
compression at every severity, but under low light with sensor noise the frozen backbone overtakes
it between severity 1 and 2 and is 16 accuracy points ahead at severity 2. Every clean-test error of
both models is a confusion between two diseases of the same crop.

Full numbers: `results/main_table.csv`, `results/final_metrics.json`,
`results/robustness/robustness_metrics.csv`. Figure: `results/robustness/degradation_curves.png`.

## Repository layout

```
1_split_test.py            carve a test split out of train
2_rename_spaces.py         sanitise directory names
3_plant_disease_resnet50.py  training, both conditions and the learning-rate ablation
4_logs_building.py         training curves and summary table from logs
5_inference.py             single-image and full-split inference, latency, per-class metrics
6_robustness_eval.py       evaluation under simulated robotic capture artifacts
7_failure_analysis.py      confusion pairs, most-confident errors, per-class fragility
8_collect_results.py       final results tree, main table, run provenance
capture_corruptions.py     corruption model shared by 6 and 7

PlantVillage/  train/ val/ test/      dataset (only test/ is committed)
checkpoints/                          model weights, published as release assets
logs/          training/ ablation_lr/ evaluation/
results/       clean/ comparison/ ablation_lr/ robustness/ failures/
report/        two-page technical summary
```

## Environment

Python 3.12, CUDA 12.9. Install with:

```
pip install -r requirements.txt
```

Training and evaluation were executed on an NVIDIA A100 80 GB PCIe through a ClearML agent using the
`nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04` image, with torch 2.10.0+cu129 and
torchvision 0.25.0+cu129. `clearml` and `boto3` are needed only for remote execution; every script
runs locally without them.

## Dataset

PlantVillage, Kaggle mirror: <https://www.kaggle.com/datasets/mohitsingh1804/plantvillage>

Download and unpack it so that `PlantVillage/train` and `PlantVillage/val` exist, then run the two
preprocessing steps **in this order**:

```
python 1_split_test.py        # moves 5% of each train class into PlantVillage/test, seed 42
python 2_rename_spaces.py     # replaces spaces in directory and file names with underscores
```

`1_split_test.py` **moves** files rather than copying them, so there is no train/test leakage — and
it is not idempotent: running it twice removes another 5% from train. Run it exactly once on a fresh
download.

Resulting split: 41 275 train / 10 861 validation / 2 169 test images over 38 classes. The validation
split ships with the Kaggle distribution; the test split is produced by step 1.

`PlantVillage/test` (2 169 images, 32 MB) is committed to this repository, so the reported numbers
can be verified without downloading the full dataset.

## Reproducing the reported results

### Verification path — no training required

Download `frozen.pth` and `finetune.pth` from the release assets into `checkpoints/`:

Release page: <https://github.com/EgorSavchenko-web/icv2026-plant-disease/releases/tag/v1.0>

```
curl -L -o checkpoints/frozen.pth   https://github.com/EgorSavchenko-web/icv2026-plant-disease/releases/download/v1.0/frozen.pth
curl -L -o checkpoints/finetune.pth https://github.com/EgorSavchenko-web/icv2026-plant-disease/releases/download/v1.0/finetune.pth
```

```
python 5_inference.py --checkpoint checkpoints/frozen.pth   --all --split test --device cuda
python 5_inference.py --checkpoint checkpoints/finetune.pth --all --split test --device cuda
python 6_robustness_eval.py --checkpoints checkpoints/frozen.pth checkpoints/finetune.pth
python 7_failure_analysis.py --grid-condition low_light:2
python 8_collect_results.py
```

This reproduces every number in the report from the committed test split. Use `--device cuda` rather
than the default `--device auto`: `auto` falls back to CPU when a GPU is not visible, which silently
invalidates the latency measurement.

Single-image prediction:

```
python 5_inference.py --checkpoint checkpoints/finetune.pth --image path/to/leaf.JPG
```

The same image can be passed through a capture artifact before prediction, which is what the demo
shows. The corruption is seeded from the image path, so a single-image run reproduces exactly the
prediction recorded for that image in `results/robustness/predictions_robustness.csv`:

```
python 5_inference.py --checkpoint checkpoints/finetune.pth \
  --image "PlantVillage/test/Tomato___Leaf_Mold/c02d931d-c724-49b8-a6c8-440c5492d747___Crnl_L.Mold_6713.JPG"

python 5_inference.py --checkpoint checkpoints/finetune.pth \
  --image "PlantVillage/test/Tomato___Leaf_Mold/c02d931d-c724-49b8-a6c8-440c5492d747___Crnl_L.Mold_6713.JPG" \
  --corrupt low_light:2 --save-corrupted demo_corrupted.jpg
```

Clean, the fine-tuned model returns Tomato Leaf Mold at confidence 1.0000. With the same leaf shot in
shade by the simulated robot, it returns Tomato Septoria leaf spot, also at confidence 1.0000.

### Full retraining

```
python 3_plant_disease_resnet50.py --mode frozen   > logs/training/frozen.log 2>&1
python 3_plant_disease_resnet50.py --mode finetune > logs/training/finetune.log 2>&1

python 3_plant_disease_resnet50.py --mode finetune --lr 1e-4 > logs/ablation_lr/lr1e-4.log 2>&1
python 3_plant_disease_resnet50.py --mode finetune --lr 1e-3 > logs/ablation_lr/lr1e-3.log 2>&1
python 3_plant_disease_resnet50.py --mode finetune --lr 1e-2 > logs/ablation_lr/lr1e-2.log 2>&1

python 4_logs_building.py --logs-dir logs/training    --out-dir results/comparison
python 4_logs_building.py --logs-dir logs/ablation_lr --out-dir results/ablation_lr
```

Each training run writes its checkpoint and artifacts to `outputs/<mode>_<timestamp>/`;
`8_collect_results.py` imports them into `results/` automatically.

Approximate cost on one A100: 9.9 min for the fine-tuned run (16 epochs, early stopping),
14.7 min for the frozen run (47 epochs). Both use batch size 64, AdamW, `ReduceLROnPlateau` on
validation macro F1, early stopping with patience 5, and seed 42.

### Remote execution on a ClearML agent

Add `--clearml --queue <queue>` to any of steps 3, 5 and 6. Checkpoints are fetched with
`--model-dataset <clearml_dataset_id>` and the dataset with `PLANTVILLAGE_DATASET_ID`. Note that a
ClearML agent packages only the entry script when no git repository is attached; `6_robustness_eval.py`
imports `capture_corruptions.py`, so remote execution requires this repository to be committed.

## Corruption model

Three independent artifacts of an autonomous field phenotyping robot, applied to the test images in
capture space, before the evaluation transform, with a per-image seed derived from the file path so
that every run is byte-identical:

| Corruption | Mechanism | Severity 1 / 2 / 3 |
|---|---|---|
| Motion blur | imaging while moving, gimbal shake | line kernel 5 / 11 / 19 px, random angle |
| Low light + sensor noise | canopy shade, overcast, dusk | exposure 0.50 / 0.30 / 0.18 |
| JPEG compression | on-board compression for radio transmission | quality 30 / 15 / 8 |

Low light is not a brightness reduction. The image is darkened, Poisson shot noise and Gaussian read
noise are applied at that reduced signal level, and the result is scaled back up, which emulates a
camera raising sensor gain: mean brightness is restored and the signal-to-noise ratio is destroyed.

## Results tree

```
results/main_table.csv          headline comparison
results/final_metrics.json      every reported number, including the shared inference cost
results/run_provenance.json     ClearML task ids, workers, container, hyperparameters, timings
results/clean/                  per-image predictions, per-class metrics, confusion matrices, curves
results/comparison/             frozen vs fine-tuned training curves
results/ablation_lr/            learning-rate ablation
results/robustness/             degradation metrics, curves, corruption examples
results/failures/               confusion pairs, most-confident errors, per-class fragility
```

Inference cost is reported once rather than per condition: both conditions are the same ResNet-50
graph with the same 23 585 894 parameters, so the forward pass is identical. Measured on the A100:
4.741 ms per image at batch 1 (210.9 img/s), 0.314 ms per image at batch 64 (3180.8 img/s),
773.1 MB peak GPU memory.

## Team

| Member | Email |
|---|---|
| Egor Savchenko | eg.savchenko@innopolis.university |
| Zamir Safin | z.safin@innopolis.university |
| Ilia Ponomarev | il.ponomarev@innopolis.university |
| Vadim Poponnikov | v.poponnikov@innopolis.university |

Individual contributions are listed in the technical summary in `report/`.
