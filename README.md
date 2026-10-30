# Traffic Sign Classifier (PyTorch)

An image classifier for German traffic signs (GTSRB, 43 classes), built in PyTorch and run on an Apple Silicon Mac (MPS). It compares a small CNN trained from scratch with a pretrained ResNet-18, and tests how accuracy holds up under blur, lighting changes, noise, haze, and rotation.

The best model reaches **98.11%** test accuracy on the official test set. Blur and noise augmentation during training makes it much more robust to noise, where the unaugmented models drop sharply.

## Results

Official GTSRB test set, 12,630 images. Validation was a random 10% of the training set, so validation scores are optimistic (see Limitations).

| Model | Test accuracy |
|---|---|
| Baseline CNN, trained from scratch (32 px) | 96.69% |
| ResNet-18, frozen backbone, 128 px | 73.00% |
| ResNet-18, frozen backbone, 224 px | 78.60% |
| ResNet-18, fine-tuned, 128 px, seed 42 | 96.37% |
| ResNet-18, fine-tuned, 128 px, seed 0 | 96.04% |
| ResNet-18, fine-tuned, 224 px, seed 42 | 97.69% |
| ResNet-18, fine-tuned, 224 px, seed 0 | 97.56% |
| **ResNet-18, fine-tuned, 224 px, with blur and noise augmentation** | **98.11%** |

Per-class accuracy and confusion matrices for each run are in `outputs/`. Every run and its settings are logged in [`docs/experiments.md`](docs/experiments.md).

### Robustness

Each corruption is applied to the test images at three strengths. Accuracy at the strongest level:

| Corruption (strongest level) | Baseline CNN | Fine-tuned 224 px | Fine-tuned 224 px + augmentation |
|---|---|---|---|
| Blur (sigma 1.5 px) | 79.53% | 51.00% | **85.30%** |
| Darker (×0.3) | 87.28% | **94.73%** | 89.79% |
| Brighter (×2.0) | 89.50% | 89.78% | **91.43%** |
| Noise (std 40) | 42.92% | 15.10% | **62.83%** |
| Fog-like haze (60% blend) | 53.69% | **83.48%** | 80.01% |
| Rotation (15°) | 87.40% | 73.80% | **91.68%** |

The augmented model beats the baseline in all 18 corruption conditions. Compared with the unaugmented fine-tuned model, it loses slightly on the two strongest darkness and haze levels.

Full tables and the plot: [`outputs/robustness_results.csv`](outputs/robustness_results.csv), [`outputs/robustness_plot.png`](outputs/robustness_plot.png).

## Setup

Requires macOS on Apple Silicon (MPS) or any machine with PyTorch. Training falls back to the CPU when MPS is not available.

```bash
conda create -n tsc python=3.11 -y
conda activate tsc
pip install -r requirements.txt
```

Check the GPU backend:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Data

The GTSRB images are not in this repo. Put the dataset in `data/gtsrb/` using torchvision's layout:

```
data/gtsrb/
  GTSRB/Training/00000/*.ppm ... 00042/*.ppm     # 39,209 training images
  GTSRB/Final_Test/Images/*.ppm                  # 12,630 test images
  GT-final_test.csv                              # test labels, ';' separated (Filename;ClassId)
```

The torchvision `GTSRB` class reads this layout. If you have the Kaggle copy of GTSRB (a `Train/` folder with 43 class folders, a `Test/` folder, and `Test.csv`) in `data/`, convert it with:

```bash
python -m tsc.prepare_data
```

The converter only reads the Kaggle files. It refuses to overwrite an existing `data/gtsrb/`.

## Running

Run from the repo root with the environment active. Each command writes its outputs to `outputs/`.

```bash
export PYTHONPATH=src

python -m tsc.train                                # baseline CNN, saves outputs/baseline.pt
python -m tsc.evaluate                             # baseline test accuracy, per-class CSV, confusion matrix

python -m tsc.transfer --size 224                  # frozen ResNet-18 head (needed before fine-tuning at 224 px)
python -m tsc.finetune --size 224                  # fine-tuned ResNet-18, seed 42
python -m tsc.finetune --size 224 --augment        # fine-tuned with blur and noise augmentation

python -m tsc.robustness                           # corrupts the test set, writes results CSV and plot
```

Each run takes between a few minutes (baseline and frozen head) and about 25 minutes (fine-tuning at 224 px). Keep the Mac plugged in and awake during long runs.

Classify your own photos. The model expects a tightly cropped sign, not a full street scene:

```bash
python -m tsc.predict my_sign.jpg --model resnet-augmented
```

Tests (no dataset or checkpoints needed):

```bash
pytest
```

## Project structure

```
src/tsc/
  data.py          dataset loading, resizing, train/validation split
  models.py        the baseline CNN
  train.py         baseline training loop, shared helpers (device, seed, normalisation)
  evaluate.py      test accuracy, per-class accuracy, confusion matrix
  transfer.py      ResNet-18 with a new head, frozen backbone
  finetune.py      unfrozen fine-tuning, optional augmentation
  robustness.py    corruptions and the robustness test
  predict.py       classify your own cropped sign photos
tests/             pytest suite (14 tests)
docs/experiments.md  every run and its settings
outputs/           results (CSV, plots); checkpoints are kept local
```

## Limitations

- **Validation is optimistic.** The validation split is a random 10% of the training set. GTSRB training images come in tracks of the same sign, so near-duplicate frames end up on both sides. The test set is the honest number.
- **Few seeds.** The 224 px fine-tuning was run with two seeds (97.69% and 97.56%). That shows the result is stable, but it's not a confidence interval.
- **Synthetic corruptions.** Blur, noise, haze, and darkness are simple approximations of real camera effects. Rotation only turns one way.
- **Augmentation tested on the same kinds of corruption.** The augmented model was trained on blur and noise, and the robustness test uses blur and noise too. Its gains on noise severities 1 and 2 are therefore inside the training range. Severity 3 is outside it, and the model still gains there.
- **Unequal training budgets.** The baseline trained for 10 epochs. The ResNet runs trained for 5 frozen epochs, then 5 fine-tuning epochs. The baseline also works at its native 32 px, while the ResNet runs upscale to 128 or 224 px.
- **Cropped signs only.** The classifier does not locate signs in a larger photo.
- **Not bit-for-bit reproducible.** MPS training can vary slightly between runs.
