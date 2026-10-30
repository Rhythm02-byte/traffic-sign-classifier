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

Per-class results are in `outputs/*_per_class_accuracy.csv`, and confusion matrices are in `outputs/*_confusion_matrix.png`, one pair per row above.

## Notes

- The validation split is a random 10% of the training set. GTSRB training images come in tracks of the same physical sign, so validation scores are optimistic. Use the test set for reporting.
- The Kaggle copy of GTSRB was converted to torchvision's layout (PNG to PPM, zero-padded folder names). Pixels are unchanged. The conversion scripts are not in the repo.
- Frozen ResNet-18 is worse than the baseline on 42 of 43 classes. Likely causes: 32 px images upscaled to 128 px, ImageNet features not adapted to signs, and only the final layer learning. Fine-tuning is the next experiment.
- The ResNet-18 input size is 128 px by default. The 224 px runs use the pretrained size and cost about three times as much compute.
- Fine-tuning at 128 px varied by 0.3 points between seeds (96.37% vs 96.04%). The 224 px fine-tuned model reaches 97.69%, 1 point above the baseline. That's more than the seed spread we've seen, but it's one seed at 224 px, so it needs a second seed to confirm.
- At 224 px the fine-tuned model beats the baseline on 24 classes and loses on 17.
- Going from 128 px to 224 px raises the frozen model by 5.6 points, and the fine-tuned model by 1.3 points over its 128 px seed-42 run.

## Next

1. Run a second seed at 224 px to confirm the 97.69% result.
2. Robustness evaluation: blur, brightness, noise, fog-like corruption, and rotation, comparing the baseline and the 224 px fine-tuned model.
