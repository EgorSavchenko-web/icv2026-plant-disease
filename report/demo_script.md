# Demo script - 5 minutes, then Q&A

Confirmed with the TA: the slot is 10 minutes and includes the Q&A; the presentation itself is 5
minutes. Three slides plus a live demo.

Presenter: Vadim Poponnikov. Rehearse end to end at least twice, on the machine you will present
from, with the repository cloned, both checkpoints downloaded, and all three commands already in the
terminal history.

## Before the session

```
git clone https://github.com/EgorSavchenko-web/icv2026-plant-disease
cd icv2026-plant-disease
pip install -r requirements.txt
mkdir checkpoints
curl -L -o checkpoints/finetune.pth     https://github.com/EgorSavchenko-web/icv2026-plant-disease/releases/download/v1.1/finetune.pth
curl -L -o checkpoints/linear_probe.pth https://github.com/EgorSavchenko-web/icv2026-plant-disease/releases/download/v1.1/linear_probe.pth
```

Large terminal font. `slides.pdf` open in a second window. Run all three commands once beforehand -
the first invocation loads torch and takes a few seconds longer than the rest.

## Timing

| Time | What | Slide |
|---|---|---|
| 0:00-0:45 | Question and setup | 1 |
| 0:45-2:00 | The result and the control | 2 |
| 2:00-3:30 | Live demo | terminal |
| 3:30-4:30 | Why it breaks, and the take-away | 3 |
| 4:30-5:00 | Buffer |  |

## 0:00-0:45, slide 1

PlantVillage: 38 plant diseases, photographed in a laboratory - one detached leaf, grey background,
even light. A pretrained ResNet-50 saturates on it, so we asked a harder question.

Same network, one variable: which parameters receive gradients. The baseline freezes the backbone
and trains only the classifier, 78 thousand parameters against 23.6 million - and it is a true
linear probe, with BatchNorm held in evaluation mode so the features really are fixed. Each
condition had its learning rate tuned separately on validation, because they solve different
optimisation problems.

Then we photograph the test set with a field robot instead of a laboratory operator: motion blur
while it drives, low light with sensor noise under the canopy, JPEG for the radio link. Three
severities, three seeds per condition, no retraining.

## 0:45-2:00, slide 2

On clean data fine-tuning wins decisively: three errors out of 2169 against thirty-four.

Under blur and JPEG it stays ahead at every severity, by up to eleven points. But look at the second
panel. Under low light with sensor noise the curves cross, and at severity two the frozen probe is
twenty-four points ahead. Every seed agrees: the best fine-tuned run still falls below the worst
probe run, so this is not seed noise.

Now the third panel, which is a control. Pure additive Gaussian noise, same brightness, no
darkening, no gain. It reproduces the reversal - and produces it one severity earlier. So noise
alone is enough; the darkness and the camera gain are not needed. What the control does not tell us
is why noise hurts the fine-tuned model - that part is still a hypothesis, and we will say so.

The effect itself matches what Kumar and colleagues reported at ICLR in 2022: fine-tuning gains in
distribution and can lose to linear probing out of it. Their explanation is different from ours, and
we test neither. What we add is that the reversal is not a property of distribution shift in
general - it depends on the kind of degradation.

## 2:00-3:30, live demo

> Say: this is not a picture of the experiment, it is the experiment. The corruption is seeded from
> the image path, so what you see now is exactly what is recorded in the results file.

```
python 5_inference.py --checkpoint checkpoints/finetune.pth \
  --image "PlantVillage/test/Apple___healthy/3aff0cab-c5f0-4fd5-8a7a-2a81160da701___RS_HL_7392.JPG"
```

Expected: `Apple___healthy  1.0000`.

> Say: a healthy apple leaf. The fine-tuned model is right, at confidence one. Now the same leaf,
> photographed by the robot in the shade.

```
python 5_inference.py --checkpoint checkpoints/finetune.pth \
  --image "PlantVillage/test/Apple___healthy/3aff0cab-c5f0-4fd5-8a7a-2a81160da701___RS_HL_7392.JPG" \
  --corrupt low_light:2 --save-corrupted demo_corrupted.jpg
```

Expected: `Tomato___Septoria_leaf_spot  0.6142`.

> Say: same leaf - and now it is a tomato disease. Not even the right plant. And this is not a
> random mistake: under this corruption between sixty-three and seventy-seven per cent of the
> fine-tuned model's errors, depending on the seed, land on Septoria leaf spot. Septoria looks like
> fine dark speckle, which is what sensor noise adds.

```
python 5_inference.py --checkpoint checkpoints/linear_probe.pth \
  --image "PlantVillage/test/Apple___healthy/3aff0cab-c5f0-4fd5-8a7a-2a81160da701___RS_HL_7392.JPG" \
  --corrupt low_light:2
```

Expected: `Apple___healthy  1.0000`.

> Say: the same degraded image, through the frozen backbone. Correct, at confidence one. One leaf is
> an illustration, not evidence, so to be clear how it was chosen: it is one of 202 test images where
> all six runs are right on the clean photo, all three probe seeds stay right in the shade and all
> three fine-tuned seeds go wrong. The reverse happens too, on 34 images. The curves are the
> evidence; this is what one of those failures looks like.

Optionally open `demo_corrupted.jpg` to show the leaf is still perfectly readable to a human.

Backup specimen if the first misbehaves, chosen by the same rule:
`Apple___Apple_scab/be517ff5-5e63-4f6a-9f55-1522c9ec972b___FREC_Scab_3123.JPG` - fine-tuned goes
from Apple scab at 1.0000 to Septoria leaf spot at 0.9008 under `--corrupt low_light:2`; the probe
stays on Apple scab at 1.0000.

## 3:30-4:30, slide 3

Why - and this is our hypothesis, not a result: fine-tuning on noise-free laboratory images may
tune the early filters to texture statistics that only that capture regime produces. Blur and JPEG
remove high-frequency content, so the adapted features still carry the signal; noise adds energy in
exactly that band. The direct test is noise of equal energy restricted to low or to high
frequencies. We have not run it.

What breaks: every clean error of the fine-tuned model, two or three per seed, is a confusion
between two diseases of the same crop. Under noise its errors pile onto one class, as we just saw.
The probe drifts toward the same speckled tomato diseases, much less strongly, so part of the pull
comes from the label set and fine-tuning amplifies it.

And calibration. On clean data the fine-tuned model is the better calibrated of the two. At low
light severity 2 that inverts: expected calibration error 0.26 to 0.42 against 0.17 to 0.20 across
the three seeds. The problem is
not that it is generally overconfident; it is that it stops signalling when it has left its
operating regime, which for anything deployed on a machine is worse than the accuracy it wins
elsewhere.

Take away: fine-tune by default, but robustness is not a single number - it flips with the kind of
degradation. Choose the transfer strategy for the capture regime you expect, and report calibration,
not only accuracy.

---

# Q&A preparation

Five of the twenty-five project points are individual. Every member should be able to answer
anything below; the name marks who must be word-perfect.

## Design and method

**Why ResNet-50 and not a ViT or something newer?** *(Ilia)* The project asks for a controlled
comparison of transfer strategies, not a backbone search. ResNet-50 with ImageNet weights is the
standard reference for PlantVillage and fits one GPU; holding it fixed is what lets us attribute the
difference to the transfer strategy. Changing the architecture would have changed two things at
once.

**Is your baseline really a linear probe?** *(Egor)* Yes, and getting that right mattered. Setting
`requires_grad = False` does not freeze BatchNorm running statistics, so a naively frozen backbone
still adapts to the new data. Our `linear_probe` mode keeps every BatchNorm module in evaluation
mode during training. We kept the naive version as an ablation: at the same learning rate it gives
66 errors against 56, so BatchNorm adaptation was mildly harmful.

**Did you tune the baseline's learning rate?** *(Egor)* Separately from fine-tuning, because they
are different optimisation problems - the probe is a convex problem on fixed features. We swept
1e-4 to 1e-1 and selected 1e-2 on validation macro F1; the optimum is interior, both neighbours are
worse. At the original 1e-4 the probe never even triggered early stopping, it was simply
undertrained, and that inflated the gap from 34 errors to 66.

**Why three seeds?** *(Zamir)* Because a single run cannot separate a real effect from
initialisation luck, and fine-tuned robustness in particular varies with the seed - our own spread
at low light severity 2 is 15 points. The conclusion holds anyway: the ranges do not overlap.

## Metrics and thresholds

**Why macro F1 rather than accuracy for selection?** *(Zamir)* 38 classes with unequal support -
macro F1 weights every class equally, so a model cannot look good by being right on the large ones.
Accuracy is reported alongside because published PlantVillage numbers use it.

**What threshold does the classifier use?** *(Egor)* None. It is a 38-way argmax over the softmax;
there is no operating threshold. The confidences we report are the softmax maximum, used as a
calibration signal, not as a decision rule.

**Why expected calibration error and not mean confidence?** *(Egor)* Because mean confidence on
errors is computed over three images when the model makes three errors, which is meaningless. ECE
bins every prediction and compares confidence with observed accuracy, so it uses all 2169.

**How were the corruption severities chosen?** *(Vadim)* So that severity 3 degrades the network
while a human can still identify the lesion. We chose them by inspecting degraded test images, which
is a mild form of test-set peeking, and we say so in the report.

## Results

**99.9 % looks too good - is there leakage?** *(Zamir)* There is a small amount and we measured it
rather than asserting its absence. PlantVillage contains several photographs of one physical leaf;
Mohanty and colleagues kept them on one side of their splits and our file-level split does not. A
perceptual hash over all 43,444 images finds one exact duplicate and 51 test images, 2.35 per
cent, within Hamming distance 5 of a training image, forty-four of them in the same class. Dropping
all fifty-one moves accuracy by at most one ten-thousandth in either direction for every run; one
linear-probe error per seed falls on a flagged image and no fine-tuned error does. It is a lower
bound, since this hash only catches near-identical framing and does not match a leaf that was
rotated or mirrored between shots.

**Why is inference latency reported once?** *(Egor)* Both conditions are the same ResNet-50 graph
with the same 23,585,894 parameters, so the forward pass is identical. We measured both on the same
A100: 7.09 against 7.30 milliseconds at batch 1, 0.355 for both at batch 64, and 773.1 megabytes of
peak memory for both. What differs is training cost.

**Is the crossover statistically meaningful?** *(Egor)* Paired McNemar on identical inputs gives
p around 1e-84 at low light severity 2 and 1e-134 at severity 3. Across seeds the ranges do not
overlap. The one cell where the two are indistinguishable is low light severity 1, p = 0.11.

## Failures

**Show a failure and explain it.** *(Vadim)* The demo case: a healthy apple leaf, correct at
confidence 1.0000 clean, classified as tomato Septoria leaf spot at 0.6142 under low light severity
2, while the probe keeps it as healthy apple. Septoria presents as fine dark speckling, which is
visually what sensor noise adds. Under Gaussian noise severity 2 the fine-tuned model puts 61 to 76
per cent of its errors on Septoria, depending on the seed; the probe's largest single label takes 22
to 40 per cent.

**Was the demo image cherry-picked?** *(Vadim)* Yes, and by a rule we state: right on the clean image
in all six runs, right under low light severity 2 for all three probe seeds, wrong for all three
fine-tuned seeds. 202 test images qualify. The reverse case exists on 34 images. The demo illustrates
the failure; the degradation curves and the paired tests are the evidence.

**Why does noise hurt the fine-tuned model?** *(Egor)* We know that noise is sufficient: the Gaussian
control reproduces the reversal without darkening or gain. We do not know how it acts. Our
hypothesis is that fine-tuning tunes early filters to high-frequency laboratory texture. The test
would be band-limited noise of equal energy, low against high frequencies, following Yin and
colleagues' Fourier analysis of robustness; if the hypothesis is right, the fine-tuned model fails on
the high band only. We have not run it, and the report calls it a hypothesis.

**Why those particular disease confusions?** *(Zamir)* All clean errors of the fine-tuned model,
two or three per seed, are within the same crop, so it never confuses the plant, only the symptom. Corn Cercospora and Northern
Leaf Blight both produce elongated grey-brown lesions, and on a leaf far into necrosis the geometry
that separates them is gone.

**What would you do next?** *(Ilia)* A leaf-grouped split, training with corruption augmentation to
see whether the reversal survives it, and evaluation on genuinely field-captured images over the
intersecting classes of a dataset such as PlantDoc - which is the limitation we could not resolve
inside PlantVillage. There is also a known background shortcut in this dataset: Noyan reaches 49 per
cent accuracy from eight background pixels alone.
