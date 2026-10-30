# Experiment log

Test set: official GTSRB test split, 12,630 images. Validation: 10% of the training set (3,920 images), fixed seed 42.

## Results so far

| Model | Trainable | Best validation | Test accuracy |
|---|---|---|---|
| Baseline CNN, from scratch (`tsc.train`, `tsc.evaluate`) | all | 0.996 | **0.9669** |
| ResNet-18, ImageNet weights, backbone frozen, new head (`tsc.transfer`) | 22,059 of 11,198,571 | 0.883 | **0.7300** |
| ResNet-18, fine-tuned, 128 px, seed 42 (`tsc.finetune`) | 11,198,571 | 0.999 | **0.9637** |
| ResNet-18, fine-tuned, 128 px, seed 0 (`tsc.finetune --seed 0`) | 11,198,571 | 0.998 | **0.9604** |
| ResNet-18, frozen backbone, 224 px (`tsc.transfer --size 224`) | 22,059 of 11,198,571 | 0.908 | **0.7860** |
| ResNet-18, fine-tuned, 224 px, seed 42 (`tsc.finetune --size 224`) | 11,198,571 | 0.999 | **0.9769** |
| ResNet-18, fine-tuned, 224 px, seed 0 (`tsc.finetune --size 224 --seed 0`) | 11,198,571 | 0.999 | **0.9756** |

Per-class results are in `outputs/*_per_class_accuracy.csv`, and confusion matrices are in `outputs/*_confusion_matrix.png`, one pair per row above.

## Notes

- The validation split is a random 10% of the training set. GTSRB training images come in tracks of the same physical sign, so validation scores are optimistic. Use the test set for reporting.
- The Kaggle copy of GTSRB was converted to torchvision's layout (PNG to PPM, zero-padded folder names). Pixels are unchanged. The conversion scripts are not in the repo.
- Frozen ResNet-18 is worse than the baseline on 42 of 43 classes. Likely causes: 32 px images upscaled to 128 px, ImageNet features not adapted to signs, and only the final layer learning. Fine-tuning is the next experiment.
- The ResNet-18 input size is 128 px by default. The 224 px runs use the pretrained size and cost about three times as much compute.
- Fine-tuning at 128 px varied by 0.3 points between seeds (96.37% vs 96.04%). The 224 px fine-tuned model reaches 97.69% (seed 42) and 97.56% (seed 0), about 1 point above the baseline. The two seeds differ by 0.13 points, so the gain holds across seeds. Both are still two runs, not a confidence interval.
- At 224 px the fine-tuned model beats the baseline on 24 classes and loses on 17.
- Going from 128 px to 224 px raises the frozen model by 5.6 points, and the fine-tuned model by 1.3 points over its 128 px seed-42 run.

## Next

1. Test whether training with blur and noise augmentation closes the gap on noisy and blurry images.
2. Add pytest tests, `predict.py`, and the README.

## Robustness (official test set, severities 1-3)

Corruptions are applied to the 32 px test images, so both models see identical inputs. Results are in `outputs/robustness_results.csv` and `outputs/robustness_plot.png`.

| Condition | Baseline CNN | ResNet-18, 224 px, fine-tuned |
|---|---|---|
| Clean | 96.69% | **97.69%** |
| Blur 1 / 2 / 3 | 96.18 / 91.22 / 79.53% | **97.66** / 87.99 / 51.00% |
| Darker 1 / 2 / 3 | 96.07 / 93.94 / 87.28% | **97.42 / 96.84 / 94.73%** |
| Brighter 1 / 2 / 3 | 95.38 / 93.37 / 89.50% | **96.11** / 93.74 / 89.78% |
| Noise 1 / 2 / 3 | **86.84 / 68.93 / 42.92%** | 64.89 / 36.42 / 15.10% |
| Fog-like haze 1 / 2 / 3 | 93.79 / 80.59 / 53.69% | **96.57 / 91.56 / 83.48%** |
| Rotation 1 / 2 / 3 | **95.06 / 92.41 / 87.40%** | 95.61 / 87.35 / 73.80% |

- The fine-tuned model is better on clean images, darkness, and haze. It is much worse on noise, and worse on heavy blur and on large rotations.
- The clean-accuracy gain does not carry over to every corruption. A single clean-set number would overstate the model.
- Single seed. The corruption strengths are simple approximations of real conditions.
