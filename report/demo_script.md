# Demo script - 5 minutes, then Q&A

Format confirmed with the TA: the slot is 10 minutes and includes the Q&A; the presentation itself
is 5 minutes. Three slides plus a live demo.

Presenter: Vadim Poponnikov. Everything below must be rehearsed end to end at least twice, with the
repository cloned, the checkpoints downloaded from the release, and the two commands already typed
into the terminal history.

## Before the session

```
git clone https://github.com/EgorSavchenko-web/icv2026-plant-disease
cd icv2026-plant-disease
pip install -r requirements.txt
mkdir checkpoints
curl -L -o checkpoints/finetune.pth https://github.com/EgorSavchenko-web/icv2026-plant-disease/releases/download/v1.0/finetune.pth
```

Open a terminal with a large font. Have `slides.pdf` open in a second window. Test both commands once
on the presentation machine - the first run loads torch and takes a few seconds longer.

## Timing

| Time | What | Slide |
|---|---|---|
| 0:00-0:45 | Problem, dataset, question | 1 |
| 0:45-2:00 | Results and the crossover | 2 |
| 2:00-3:30 | Live demo | terminal |
| 3:30-4:30 | Failure analysis and conclusion | 3 |
| 4:30-5:00 | Buffer |  |

## 0:00-0:45, slide 1

PlantVillage: 38 disease classes over 14 crops, photographed in a laboratory - one detached leaf,
uniform background, even light. 41 275 training, 10 861 validation, 2 169 test images; the test split
was carved by moving five per cent of each training class out of train, so no image is in two splits.

We compare two transfer strategies on one pretrained ResNet-50. The baseline freezes the backbone and
trains only the classifier, 77 862 parameters. The second condition fine-tunes all 23.6 million. That
is the only thing that differs between the two runs.

A pretrained network saturates on clean PlantVillage, so the interesting question is not whether
transfer learning works. It is which strategy survives leaving the laboratory. We model a field
phenotyping robot: motion blur while it drives, low light with sensor noise under the canopy, and
JPEG compression before radio transmission - three severities each, applied to the test set only,
with no retraining.

## 0:45-2:00, slide 2

On clean data fine-tuning wins decisively: three errors out of 2 169 against sixty-six, and it gets
there in sixteen epochs instead of forty-seven.

Now the degradation curves. Under motion blur and under JPEG compression fine-tuning stays ahead at
every severity, by up to twenty-three accuracy points. But look at the middle panel. Under low light
with sensor noise the curves cross between severity one and two, and the frozen backbone ends up
sixteen points ahead at severity two and seventeen at severity three.

Our reading is a frequency-domain one. Fine-tuning on noise-free laboratory images sharpens the early
filters onto high-frequency texture statistics that only that capture regime produces. Blur and JPEG
remove or quantise high frequencies, so the adapted features still carry the signal. Sensor noise
injects random energy into exactly the band those filters occupy, and they have nothing generic left
to fall back on. The frozen ImageNet backbone was trained on web photographs that already contain
noise and compression, so it keeps a tolerance that fine-tuning trades away.

The learning-rate ablation confirms this is not an artefact of the learning rate: all three values
land within 0.3 accuracy points of each other.

## 2:00-3:30, live demo

> Say: this is not a picture of the experiment, it is the experiment. The corruption is seeded from
> the image path, so what you see now is the same prediction recorded in the results CSV.

```
python 5_inference.py --checkpoint checkpoints/finetune.pth \
  --image "PlantVillage/test/Tomato___Leaf_Mold/c02d931d-c724-49b8-a6c8-440c5492d747___Crnl_L.Mold_6713.JPG"
```

Expected: `Tomato___Leaf_Mold  1.0000`.

> Say: a tomato leaf with leaf mold. The model is right, at confidence one point zero. Now the same
> leaf, photographed by the robot in the shade under the canopy.

```
python 5_inference.py --checkpoint checkpoints/finetune.pth \
  --image "PlantVillage/test/Tomato___Leaf_Mold/c02d931d-c724-49b8-a6c8-440c5492d747___Crnl_L.Mold_6713.JPG" \
  --corrupt low_light:2 --save-corrupted demo_corrupted.jpg
```

Expected: `Tomato___Septoria_leaf_spot  1.0000`.

> Say: the same leaf. The model now says Septoria leaf spot, and it is again at confidence one point
> zero. It is not hesitating. It is wrong and certain. That is the finding that matters for anyone
> who would deploy this on a machine.

Optionally open `demo_corrupted.jpg` to show that the leaf is still perfectly readable to a human.

Backup specimen if something goes wrong with the first:
`PlantVillage/test/Squash___Powdery_mildew/39246e21-924e-40d6-9013-a2ca0636eb0e___MD_Powd.M_0965.JPG`
with `--corrupt motion_blur:2` - squash powdery mildew becomes healthy maize at 0.9999.

## 3:30-4:30, slide 3

Every clean-test error of both models is a confusion between two diseases of the same crop - all
three of the fine-tuned model and all sixty-six of the frozen one. The cross-crop error rate is
exactly zero. That answers the question the project brief asked us to investigate.

The calibration gap is the part we did not expect. Mean confidence on errors is 0.86 for the
fine-tuned model against 0.53 for the frozen one, and under motion blur severity three it still
reports 0.86 while being right 58.5 per cent of the time.

Which classes fail first is not random either: those whose evidence is fine-grained. At low light
severity two, Potato healthy, Cedar apple rust and Tomato Bacterial spot fall from an F1 near one to
zero. Healthy classes collapse because "no lesion" becomes indistinguishable from "lesion hidden by
noise".

Conclusion: fine-tuning is the right default, but robustness is artifact-dependent and is not a
single number. Choose the transfer strategy for the capture regime you expect, and report
calibration, not only accuracy.

---

# Q&A preparation

Five of the twenty-five project points are individual. Each member is asked to explain a model
choice, a metric, a result, a threshold or a failure case. Every member should be able to answer
anything below; the owner column says who must be word-perfect.

## Model and design

**Why ResNet-50 and not a ViT or a newer CNN?** *(Ilia)* The project asks for a controlled
comparison of transfer strategies, not a search for the strongest backbone. ResNet-50 with ImageNet
weights is the standard reference for PlantVillage, it fits a single GPU comfortably, and holding it
fixed is what lets us attribute the difference to the transfer strategy alone. Swapping the
architecture would have changed two things at once.

**Why is a frozen backbone a fair baseline?** *(Ilia)* It is the cheapest sensible transfer strategy
and the one you would actually pick under a tight compute or data budget - 77 862 trainable
parameters against 23.6 million. It is also the standard linear-probe protocol for measuring how good
pretrained features already are.

**Why not train from scratch?** *(Ilia)* The guidelines encourage pretrained models, and from-scratch
training on 41 000 images would confound the comparison with an optimisation problem.

**Why no augmentation matching the corruptions?** *(Egor)* That would answer a different question.
We measure robustness as a property of the representation learned from clean data. Training on
corrupted data is the obvious follow-up, and we name it as such.

## Metrics and thresholds

**Why macro F1 rather than accuracy for model selection?** *(Zamir)* 38 classes with unequal support.
Macro F1 weights every class equally, so a model cannot look good by being right on the large classes
while failing the small ones. Accuracy is reported alongside because it is what published PlantVillage
numbers use.

**What threshold does the classifier use?** *(Egor)* None. It is 38-way argmax over the softmax; there
is no operating threshold to tune. The confidence numbers we report are the softmax maximum, used as
a calibration signal, not as a decision rule.

**Early stopping: what is the criterion?** *(Egor)* Validation macro F1, patience 5, maximum 50
epochs, with `ReduceLROnPlateau` on the same quantity. The frozen run stopped at epoch 47 with its
best at 42; the fine-tuned one stopped at 16 with its best at 11.

**Why three severities and those particular values?** *(Egor)* They were calibrated so that severity 3
degrades the network while a human reader can still identify the lesion - that is the regime where a
failure is informative. Severity 1 is deliberately mild to give the curve a near-clean anchor.

## Results

**99.86 per cent looks too good. Is there leakage?** *(Zamir)* No leakage by construction: the test
split is built by moving files out of train, never copying, so an image cannot be in two splits, and
we assert the class lists match across splits. The high number reflects that PlantVillage is
laboratory-captured and contains several images of the same physical leaf - we state that as a
limitation, and it is exactly why we added the degradation study.

**Why is the frozen model faster per epoch but slower overall?** *(Egor)* It needs far more epochs to
converge - 47 against 16 - because 77 862 parameters on unadapted features is a much harder
optimisation problem. Total training time 14.7 minutes against 9.9.

**Why is inference latency reported once instead of per model?** *(Egor)* Both conditions are the same
ResNet-50 graph with the same 23 585 894 parameters. The forward pass is identical, so latency and
inference memory are a property of the architecture. Measuring them twice would be measuring the same
quantity twice. What differs is training cost, which we report separately.

**Is the crossover statistically meaningful or noise?** *(Egor)* It is 16 accuracy points on 2 169
images at severity 2 and 17 points at severity 3, in the same direction at both severities, with the
opposite ordering at every severity of the other two artifacts. It is far outside anything sampling
noise on this test set could produce.

## Failures

**Show us a failure and explain it.** *(Vadim)* The demo case: Tomato Leaf Mold, correct at confidence
1.0000 on the clean image, classified as Tomato Septoria leaf spot at confidence 1.0000 under low
light severity 2. Both are tomato foliar diseases whose discriminating evidence is fine speckling;
sensor noise adds speckle-like high-frequency structure, and the model reads the noise as the wrong
symptom.

**Why do the models confuse those particular diseases?** *(Zamir)* All errors are within the same
crop, so the model never confuses the plant - it confuses the symptom. Corn Cercospora and Northern
Leaf Blight both produce elongated grey-brown lesions; on a leaf far into necrosis, the lesion
geometry that separates them is gone.

**What would you do next?** *(Ilia)* Train with corruption augmentation and re-measure the crossover;
add focus and perspective artifacts; and validate on genuinely field-captured images with a matched
class inventory, which is the limitation we could not resolve within this dataset.
