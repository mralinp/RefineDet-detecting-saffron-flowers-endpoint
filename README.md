# saffron-cut

Detects saffron flowers in an image and, for each one, predicts its
**center (x, y)** and **cutting angle** (0-360 degrees) -- the course
project's reframing of object detection: instead of a bounding box per
object, we want a center point and an orientation. The intended end use is
a robotic arm that harvests saffron flowers, using this center+angle
output to know where and at what angle to cut.

The model is a from-scratch **PyTorch reimplementation of RefineDet**
(Zhang et al., CVPR 2018, "Single-Shot Refinement Neural Network for
Object Detection"), with its box-regression heads replaced by center and
angle regression heads. The `RefineDet/` directory is the paper's original
Caffe C++ release, kept as an architectural reference; it is not built or
run, since it targets a Caffe/CUDA toolchain this project doesn't depend
on -- see `report/technical_report.md` for the rationale.

## Layout

```
main.py               entry point: trains, then predicts on Test/
configs/default.yaml   all hyperparameters
src/saffron_cut/
  config.py             Config dataclass
  device.py              cuda -> mps -> cpu selection
  data/                   dataset, augmentation, CSV I/O
  model/                  VGG16 backbone, ARM/TCB/ODM heads, loss, anchors
  engine/                 train / evaluate / predict / semi-supervised
  utils/                  seeding, visualization
data/                   Labeled/, Unlabeled/, Test/ (gitignored; see below)
RefineDet/              original Caffe repo, reference only
report/                 technical report + figures
```

## Data

Tracked in git under `data/` (so `git clone` alone is enough on the GPU VM
/ Colab). Expected layout, either directly next to `main.py`, or under
`./data/` as in this repo -- both are auto-detected, see
`resolve_data_root` in `main.py`:

- `Labeled/NNN.jpg` + `Labeled/NNN.csv` (`x, y, angle_deg` per flower, no header)
- `Unlabeled/NNN.jpg` -- unlabeled images, used for self-training
- `Test/NNN.jpg` -- running `main.py` writes `Test/NNN.csv` (`x, y, angle_deg, probability`)

`checkpoints/` (trained weights) stays gitignored -- move those between
machines with `rsync`/`scp`, not git.

## Setup

```bash
uv sync                       # or: pip install -r requirements.txt
```

On the GPU VM / Colab, install a CUDA build of torch first (see comment at
the top of `requirements.txt`); the exact same code then trains on CUDA
with no changes -- `saffron_cut.device.get_device()` picks CUDA whenever
`torch.cuda.is_available()`, and only falls back to Apple Silicon's MPS
(this dev machine) or CPU otherwise.

```bash
uv run python -c "from saffron_cut.device import device_report; print(device_report())"
```

## Running

```bash
python main.py                                 # full run: train (+ self-train on Unlabeled/) then predict Test/
python main.py --no-semi-supervised             # supervised-only baseline, skips the Unlabeled/ bonus stage
python main.py --set epochs=5 batch_size=2       # quick smoke test
python main.py --mode predict --checkpoint checkpoints/best.pt   # reuse a trained model (checkpoints/round0/best.pt if trained with self-training)
```

Checkpoints and per-epoch loss/AP history land in `checkpoints/` (see
`configs/default.yaml: checkpoint_dir`).

## Watching training live

Training logs per-iteration losses and, every `cfg.eval_every` epochs, a
validation AP/precision/recall and a rendered ground-truth-vs-prediction
image, to TensorBoard (`checkpoints/tb/`). While `main.py` is running (or
after it finishes):

```bash
uv run tensorboard --logdir checkpoints
# open http://localhost:6006 -- SCALARS for the loss/AP curves, IMAGES for the GT-vs-prediction snapshots
```

Same on the GPU VM / Colab -- just point `--logdir` at wherever that run's
`checkpoints/` ended up.

## Method summary

- **Backbone**: VGG16, ImageNet-pretrained (`torchvision`), converted to
  the atrous "fully convolutional reduced VGGNet" from the SSD/RefineDet
  papers (pool5 widened to stride 1, fc6/fc7 turned into dilated convs,
  initialized from VGG's classifier weights by kernel decimation).
- **Detection sources**: conv4_3, fc7, conv6_2, conv7_2 -- strides 8/16/32/64.
- **ARM** (Anchor Refinement Module): per-anchor objectness + a 2-D center
  offset (not a 4-D box offset -- there is no box to regress).
- **TCB** (Transfer Connection Blocks): top-down feature fusion from the
  coarsest detection source down to the finest, same role as an FPN.
- **ODM** (Object Detection Module): a second objectness + center-offset
  refinement on the TCB features, plus a third head regressing
  `(cos angle, sin angle)`, decoded with `atan2` so there's no
  discontinuity at the 0/360 wraparound.
- **Anchor matching**: since labels are points, not boxes, an anchor is
  "positive" for a flower when the anchor's own center falls within a
  small, absolute-pixel-capped radius of that flower (the point-detection
  analogue of IoU matching), plus every flower's single nearest anchor is
  always forced positive. The cap matters -- see
  `report/technical_report.md` Section 4.4 for a real bug this caught.
- **Losses**: 2-way cross-entropy with hard-negative mining (5:1) for both
  ARM and ODM, ARM's "negative anchor filtering" (RefineDet's cascade
  trick: anchors ARM is already very confident are background are dropped
  from the ODM loss), SmoothL1 on both center offsets, SmoothL1 on the
  angle vector (upweighted 3x -- it gets far fewer gradient updates than
  the classifier).
- **Semi-supervised bonus stage**: self-training over `Unlabeled/` --
  pseudo-label with the current model above a confidence threshold, mix
  those (down-weighted) into the next training round. See
  `engine/semi_supervised.py`.

Full derivation, design tradeoffs and results: `report/technical_report.md`.
