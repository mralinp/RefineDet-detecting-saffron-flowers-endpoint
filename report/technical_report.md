# Detecting the Center and Cutting Angle of Saffron Flowers with a Center/Angle-Adapted RefineDet

**Course project, Digital Image Processing, Iran University of Science and Technology.**
Base paper: Zhang, Wen, Bian, Lei & Li, *"Single-Shot Refinement Neural Network for Object Detection"*, CVPR 2018 [1].

---

## 1. Problem statement

Standard object detectors output an axis-aligned bounding box per object. This
project's target quantity is different: for every saffron flower visible in
an image, we need its **center** `(x, y)` and its **cutting angle** `theta in
[0, 360)` degrees — no box size is ever labeled or evaluated. The project
brief is explicit about this reframing: *"object detection algorithms look
for the bounding rectangle of each object, but here we are looking for the
center and angle of objects"* [9]. Concretely:

- **Input**: a 1296x972 RGB photograph of harvested saffron flowers scattered
  on a surface (`data/Labeled`, `data/Unlabeled`, `data/Test`).
- **Output**: a variable-length list of `(x, y, angle_deg)` triples, one per
  flower, plus a confidence score for the graded `Test/` predictions.
- **Supervision**: 14 labeled images (764 flowers total, `data/Labeled/*.csv`),
  129 unlabeled images (`data/Unlabeled/`) usable for semi-/self-supervised
  learning, and 18 held-out test images (`data/Test/`) that this project
  predicts on but never gets ground truth for.

## 2. Chosen paper: RefineDet

RefineDet [1] is a single-stage detector built around two cooperating
modules and a feature-fusion bridge between them:

- **ARM (Anchor Refinement Module)**: a light head on top of each of several
  backbone feature maps that (a) does *binary* objectness classification
  (background vs. any object) per anchor and (b) regresses a first-pass
  box offset. Its role is to filter the overwhelming majority of easy
  background anchors and coarsely relocate the surviving ones — the same
  problem two-stage detectors solve with a Region Proposal Network, but
  without a separate cropping/pooling stage.
- **TCB (Transfer Connection Block)**: a top-down path that fuses each
  backbone feature map with the (upsampled) richer semantic features from
  the next coarser scale, structurally the same idea as a Feature Pyramid
  Network [5], predating FPN's popularization by a few months.
- **ODM (Object Detection Module)**: the final multi-class classification
  and second-stage box regression, applied to the TCB's fused features
  using the ARM-refined anchors as its reference boxes instead of the
  original fixed anchor grid.
- **Negative anchor filtering**: anchors ARM is already highly confident
  (>0.99) are background get dropped from the ODM loss entirely, so ODM's
  training signal is concentrated on positives and genuinely ambiguous
  negatives.

The backbone is a "fully convolutional reduced" VGG16 [2, 3]: `pool5`
widened from 2x2/stride-2 to 3x3/stride-1, and the classifier's `fc6`/`fc7`
converted into a dilated 3x3 and a 1x1 convolution respectively, both
initialized from the pretrained classifier weights by kernel *decimation*
(subsampling every k-th row/column, first introduced for this exact
purpose in SSD [2]) rather than random initialization. Two extra
stride-2 conv blocks are appended after `fc7` to reach lower-resolution,
larger-receptive-field feature maps. Four feature maps in total feed the
ARM/ODM: `conv4_3` (stride 8), `fc7` (stride 16), `conv6_2` (stride 32),
`conv7_2` (stride 64). Because `conv4_3`'s activations are much larger in
magnitude than the deeper layers, it is L2-normalized with a learnable
per-channel scale before use [2].

## 3. Why a PyTorch reimplementation instead of the bundled Caffe code

`RefineDet/` in this repository is the paper authors' original Caffe C++
release. It was kept as an architectural reference but is **not built or
run**, for three concrete reasons:

1. **Target hardware.** Development happens on Apple Silicon; the actual
   training runs happen on a separate GPU VM and/or Google Colab. A
   from-scratch PyTorch model uses the exact same code path on all three —
   `torch.cuda.is_available()` selects CUDA when present and falls back to
   Apple's MPS backend otherwise (`saffron_cut/device.py`) — whereas Caffe's
   build and its CUDA/cuDNN bindings are a second, platform-specific
   toolchain with no equivalent Apple Silicon backend at all.
2. **The output heads have to be rewritten regardless.** Every one of
   RefineDet's four detection heads regresses a 4-D box offset. This
   project regresses a 2-D center offset plus a 2-D angle vector instead
   (Section 4.2) — a structural change to the prediction heads and the
   loss, not a hyperparameter. Whatever the original Caffe VOC/COCO
   checkpoint learned for 4-D box regression is not transferable to a
   2-D center + angle head, so there is no pretrained-*detector* weight to
   lose by not using the Caffe build; only the backbone's ImageNet weights
   matter, and those are available directly from `torchvision` regardless
   of framework (Section 4.1).
3. **Toolchain risk.** Caffe's build (BLAS/protobuf/cuDNN version pinning,
   Python-2-era bindings) is unmaintained and brittle to build from source
   in 2026, with no upside here since (1) and (2) already rule out reusing
   its trained weights or its Apple-Silicon-incompatible runtime.

The architecture below is therefore a faithful PyTorch reimplementation of
RefineDet's ARM/TCB/ODM cascade and its VGG16-atrous backbone, with the
prediction heads and matching/loss adapted from box regression to
center+angle regression.

## 4. Architecture adaptation

### 4.1 Backbone

`torchvision`'s ImageNet-pretrained VGG16 [3, 4] up through `relu4_3` and
`relu5_3`, with `pool5`, `fc6`, `fc7` and the two extra conv blocks built
exactly as described in Section 2 (`src/saffron_cut/model/backbone.py`).
This *is* "reaching the pretrained base model": the backbone starts from
weights trained on 1.28M ImageNet images rather than from scratch, which
matters enormously given the target dataset here has only 11-14 labeled
images (Section 6).

### 4.2 From boxes to points: anchors, heads and matching

A box detector anchors several box *shapes* per spatial cell, because it
needs to cover a range of object sizes/aspect ratios and regress toward
one of them. A point has no shape, so there is exactly **one anchor per
cell** here — just that cell's own receptive-field center
(`model/anchors.py`). Every ARM/ODM head regresses a 2-D `(dx, dy)` center
offset instead of a 4-D box offset, and the ODM head gains a third,
2-D output for the cutting angle.

**Angle encoding.** The angle is *not* regressed as a raw degree value.
The head outputs an unnormalized 2-vector trained toward
`(cos(theta), sin(theta))` with SmoothL1 loss, and is decoded with
`atan2`. This avoids the discontinuity a raw-degree regression would hit
at the 0/360 wraparound (a flower at 359 degrees and one at 1 degree are
visually near-identical but 358 degrees apart numerically) — a standard
trick for regressing circular quantities.

**Matching (anchor <-> ground truth).** SSD/RefineDet assign anchors to
ground-truth boxes by IoU. There is no ground-truth box here, so an anchor
is a positive match for a flower when the anchor's own center falls within
one grid cell of that flower's center — the direct point-detection
analogue of IoU matching, in the spirit of the center-sampling idea later
formalized by FCOS [8]. Every flower is additionally guaranteed its single
closest anchor as a positive match (mirroring SSD's "ensure every ground
truth has >=1 prior" rule), so a flower can never be orphaned even if it
falls in a gap between anchor centers (`model/matching.py`).

**Two-stage cascade, preserved.** ODM's positive/negative assignment is
recomputed against the ARM-*refined* centers (`anchor + arm_loc`), not the
raw anchor grid — the actual mechanism that makes this a cascade rather
than two independent detectors, and it is why ARM's loc loss uses
`.detach()`-ed values when feeding ODM's matching (the refinement target
gradients flow through ARM's own loss, not through ODM's).

### 4.3 Losses

Both ARM and ODM use 2-way (background/flower) softmax cross-entropy with
hard-negative mining at a 3:1 negative:positive ratio (mining the
*highest-loss* negatives, standard SSD-style practice), plus SmoothL1 on
the center offset for positive anchors. ODM additionally applies
SmoothL1 on the `(cos, sin)` angle vector for positive anchors, and its
cross-entropy is computed only over anchors that survive **negative anchor
filtering**: negatives ARM already scores as background with >0.99
confidence (or whose ARM-predicted shift is implausibly large) are
excluded from the ODM loss entirely (`model/loss.py`).

## 5. Data pipeline

### 5.1 Label format

`Labeled/NNN.csv` has one row per flower: `x, y, angle_deg` (no header,
angle spans the full `[0, 360)` circle — confirmed empirically from the
provided data, not just `[0, 180)`, so flower orientation is a *direction*,
not an undirected line). `Test/NNN.csv` is written in the same layout plus
a trailing `probability` column, per the brief's spec.

### 5.2 Augmentation, and why angle correctness is the one thing that had to be gotten exactly right

Every geometric augmentation (`data/transforms.py`) — random crop, random
rotation up to +/-20 degrees, horizontal/vertical flip, and the final
isotropic letterbox resize to the network's input resolution — transforms
flower **positions as points** (apply the affine map's linear part and
translation) and flower **orientation as a unit direction vector**
`(cos theta, sin theta)` (apply the *linear part only*, no translation),
then re-derives the angle with `atan2` after each transform. This sidesteps
manually re-deriving a sign convention per augmentation (e.g. "does a
horizontal flip mean `theta -> 180 - theta` or `theta -> -theta`?"); as
long as the point and the direction vector are pushed through the *same*
linear map, the recovered angle is correct by construction, regardless of
the coordinate system's handedness.

The other detail that would otherwise silently corrupt every angle label:
the source images are 1296x972 (4:3), not square, and the network input is
square. Resizing 1296x972 -> 512x512 **anisotropically** (independent x/y
scale factors) changes the *apparent* angle of every non-axis-aligned line
segment. All resizing in this pipeline is therefore **isotropic** —
`letterbox()` scales by `min(size/w, size/h)` uniformly and pads the
remainder — which leaves angles exactly unchanged, at the cost of some
unused padding for non-square crops.

### 5.3 Train/val split

With only 14 labeled images, `cfg.val_split` (default 3) images are held
out for validation, seeded and shuffled once (`cfg.val_seed`) so the split
is reproducible; the remaining 11 train. This is a small enough validation
set that its AP estimate should be read as indicative, not precise — see
Section 9's limitations discussion.

## 6. Training setup

SGD with momentum 0.9, weight decay 5e-4, initial LR 1e-3, linear warmup,
then a 0.1x step decay at configured epoch milestones — the same recipe
family as the original RefineDet/SSD papers. The one departure from
"textbook" VOC/COCO-scale numbers: with only 11 training images at batch
size 4, an epoch is **3 iterations**, so warmup and milestones are
expressed in small epoch/iteration counts (`cfg.lr_warmup_iters`,
`cfg.lr_milestones`) rather than the hundreds-to-thousands typical of
datasets with far more iterations per epoch; using VOC-scale warmup
counts here would leave the whole run stuck in warmup (an early run in
development did exactly this — see `configs/default.yaml`'s inline note).

Heavy geometric + photometric augmentation (Section 5.2) is the main lever
against overfitting on such a small labeled set, alongside starting from
an ImageNet-pretrained backbone rather than random initialization.

## 7. Evaluation metric

The brief specifies mAP as the grading metric but, as in training, there
is no ground-truth box to compute IoU against. This project defines a
direct analogue (`engine/evaluate.py`): a predicted flower is a true
positive against a still-unmatched ground-truth flower in the same image
when **both**

- center distance < `cfg.eval_dist_thresh` (20 px, original-image space), and
- circular angle difference < `cfg.eval_angle_thresh` (30 degrees)

hold, matched greedily in descending confidence order (each ground-truth
flower usable at most once) — the standard COCO/VOC AP matching protocol
[6], just with a distance+angle gate standing in for IoU. AP is the area
under the resulting precision-recall curve via the PASCAL VOC all-points
interpolation [6]. This is used for model selection and the results in
Section 9; the actual grading presumably uses whatever concrete mAP
definition the course staff implements over the `Test/` submissions, which
may differ in threshold or matching details from this internal metric.

## 8. Semi-supervised bonus stage: self-training

`Unlabeled/`'s 129 images are used via **self-training / pseudo-labeling**
[7] (`engine/semi_supervised.py`): after the supervised baseline (round 0)
converges, the current model runs inference over `Unlabeled/`, keeps only
detections above `cfg.pseudo_label_conf_thresh` (default 0.8) as pseudo
ground truth, and continues training on `Labeled/` + these pseudo-labeled
images together for another `cfg.pseudo_label_epochs_per_round` epochs.
Pseudo-labeled samples are **down-weighted**
(`cfg.pseudo_label_loss_weight`, default 0.5) in the loss, since they are
the model's own (imperfect) predictions rather than ground truth, and the
whole process repeats for `cfg.pseudo_label_rounds` rounds, each starting
from the previous round's weights.

This is the simplest member of the semi-supervised family the brief
suggests — plain bootstrapping, not a consistency-regularization method
(e.g. mean-teacher-style augmentation consistency) or a self-supervised
pretext task (e.g. rotation prediction, contrastive pretraining on
`Unlabeled/` before ever touching labels). It was chosen because it
reuses the existing supervised training loop and inference path almost
unchanged, which matters given the very limited compute/time budget for
this project. Its known failure mode is **confirmation bias**: if round 0
is confidently wrong about some region of image-space (e.g. a background
texture consistently scored as a flower), self-training reinforces that
mistake rather than correcting it, and the effect can compound across
rounds. This is exactly why the confidence threshold is fairly high (0.8)
and only a couple of rounds are run.

## 9. Results

*(Filled in from the actual training runs — see `checkpoints/*/history.json`
for the full per-epoch curves this section summarizes.)*

**Environment.** Numbers below are from `saffron_cut.device.device_report()`
run at the top of `main.py`, so the run's actual device is always in the
log rather than assumed:

```
<<DEVICE_REPORT>>
```

### 9.1 Supervised baseline (`--no-semi-supervised`)

<<BASELINE_RESULTS_TABLE>>

<<BASELINE_QUALITATIVE_FIGURE>>

### 9.2 Self-training on `Unlabeled/`

<<SEMI_SUPERVISED_RESULTS_TABLE>>

### 9.3 Discussion

<<RESULTS_DISCUSSION>>

## 10. Limitations and what a longer compute budget would change

- **14 labeled images is a very small dataset** for fine-tuning a
  20M-parameter detector; the 3-image validation split's AP has high
  variance, and the headline numbers here should be read as a proof that
  the pipeline works end-to-end rather than a tight estimate of true
  generalization. A production version of this project would want
  k-fold cross-validation over the labeled set rather than one fixed split.
- **The synthetic-box-free matching radius** (`cfg.pos_radius_cells`) is a
  single global hyperparameter; a per-flower adaptive radius (e.g. derived
  from local flower density) was considered but not implemented, given the
  time budget.
- **Self-training vs. stronger semi-/self-supervised methods.** A
  consistency-regularization approach (forcing agreement between the
  model's predictions on two different augmentations of the same
  unlabeled image) or self-supervised backbone pretraining on
  `Unlabeled/` before fine-tuning would likely make better use of the 129
  unlabeled images than one-shot pseudo-labeling, at the cost of
  meaningfully more implementation and compute.
- **Speed.** No inference-time optimization (TensorRT/ONNX export,
  half precision, batching beyond the default) was attempted; the
  reported inference times (Section 9) are plain PyTorch eager execution.

## 11. Reproducing this project

```bash
uv sync   # or: pip install -r requirements.txt
uv run python -c "from saffron_cut.device import device_report; print(device_report())"

python main.py                       # full run: train (+ self-train) then predict Test/
python main.py --no-semi-supervised  # supervised baseline only
```

See [`README.md`](../README.md) for the full CLI, and
[`scripts/GPU_VM.md`](../scripts/GPU_VM.md) /
[`scripts/colab_train.ipynb`](../scripts/colab_train.ipynb) for running the
identical code on CUDA.

## References

[1] S. Zhang, L. Wen, X. Bian, Z. Lei, S. Z. Li. "Single-Shot Refinement Neural Network for Object Detection." *CVPR*, 2018.

[2] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, A. C. Berg. "SSD: Single Shot MultiBox Detector." *ECCV*, 2016.

[3] K. Simonyan, A. Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*, 2015.

[4] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei. "ImageNet: A Large-Scale Hierarchical Image Database." *CVPR*, 2009.

[5] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, S. Belongie. "Feature Pyramid Networks for Object Detection." *CVPR*, 2017.

[6] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. "The PASCAL Visual Object Classes (VOC) Challenge." *IJCV*, 2010.

[7] D.-H. Lee. "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks." *ICML Workshop on Challenges in Representation Learning*, 2013.

[8] Z. Tian, C. Shen, H. Chen, T. He. "FCOS: Fully Convolutional One-Stage Object Detection." *ICCV*, 2019.

[9] Course project brief, `description.pdf`, Digital Image Processing course, Iran University of Science and Technology.

[10] A. Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*, 2019. Backbone weights via `torchvision`'s `VGG16_Weights.IMAGENET1K_V1`.
