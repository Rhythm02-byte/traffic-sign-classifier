# Experiment log

Test set: official GTSRB test split, 12,630 images. Validation: 10% of the training set (3,920 images), fixed seed 42.

## Results so far

| Model | Trainable | Best validation | Test accuracy |
|---|---|---|---|
| Baseline CNN, from scratch (`tsc.train`, `tsc.evaluate`) | all | 0.996 | **0.9669** |
| ResNet-18, ImageNet weights, backbone frozen, new head (`tsc.transfer`) | 22,059 of 11,198,571 | 0.883 | **0.7300** |

Per-class results are in `outputs/per_class_accuracy.csv` (baseline) and `outputs/resnet18_frozen_per_class_accuracy.csv`. Confusion matrices are in `outputs/confusion_matrix.png` and `outputs/resnet18_frozen_confusion_matrix.png`.

## Notes

- The validation split is a random 10% of the training set. GTSRB training images come in tracks of the same physical sign, so validation scores are optimistic. Use the test set for reporting.
- The Kaggle copy of GTSRB was converted to torchvision's layout (PNG to PPM, zero-padded folder names). Pixels are unchanged. The conversion scripts are not in the repo.
- Frozen ResNet-18 is worse than the baseline on 42 of 43 classes. Likely causes: 32 px images upscaled to 128 px, ImageNet features not adapted to signs, and only the final layer learning. Fine-tuning is the next experiment.
- The ResNet-18 input size is 128 px. The default ImageNet size is 224 px, which costs about three times as much compute.

## Next

1. Fine-tune the ResNet-18 backbone with a lower learning rate, and compare it with the frozen and baseline models.
2. Robustness evaluation: blur, brightness, noise, fog-like corruption, and rotation.
