"""Robustness: how much does accuracy drop when test images get worse?

We corrupt the official test images in controlled ways and measure both models on the same corrupted copies.
"""
import math
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import functional as TF

from tsc.data import load_test_split, to_tensors
from tsc.evaluate import collect_predictions
from tsc.models import SmallCNN
from tsc.train import BATCH_SIZE, get_device, normalize
from tsc.transfer import build_resnet18, preprocess_for_resnet

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"

# Strength of each corruption at severity 1, 2 and 3. Severity 0 means the clean image.
SEVERITY = {
    "blur": {1: 0.5, 2: 1.0, 3: 1.5},  # Gaussian blur, sigma in pixels
    "darker": {1: 0.7, 2: 0.5, 3: 0.3},  # brightness multiplier below 1
    "brighter": {1: 1.3, 2: 1.6, 3: 2.0},  # brightness multiplier above 1
    "noise": {1: 10, 2: 20, 3: 40},  # Gaussian noise, standard deviation on the 0-255 scale
    "fog": {1: 0.2, 2: 0.4, 3: 0.6},  # blend toward white, like haze
    "rotation": {1: 5, 2: 10, 3: 15},  # rotation in degrees
}


def corrupt(X, name, level, seed=0):
    # X holds uint8 images (N, 3, 32, 32). Returns corrupted uint8 images of the same shape.
    x = X.float()
    s = SEVERITY[name][level]
    if name == "blur":
        k = 2 * math.ceil(3 * s) + 1  # kernel wide enough to hold the blur
        x = TF.gaussian_blur(x, kernel_size=[k, k], sigma=[s, s])
    elif name in ("darker", "brighter"):
        x = x * s
    elif name == "noise":
        generator = torch.Generator().manual_seed(seed)  # same noise every run
        x = x + torch.randn(x.shape, generator=generator) * s
    elif name == "fog":
        x = x * (1 - s) + 255 * s
    elif name == "rotation":
        x = TF.rotate(x, angle=s)  # rotates around the centre, fills the corners with black
    return x.clamp(0, 255).round().to(torch.uint8)


def load_models(device):
    baseline = SmallCNN().to(device)
    baseline.load_state_dict(torch.load(OUTPUTS_DIR / "baseline.pt", map_location=device))

    models = {"baseline CNN": (baseline, normalize)}
    for name, checkpoint in [
        ("ResNet-18 224px fine-tuned", "resnet18_finetuned_224px.pt"),
        ("ResNet-18 224px augmented", "resnet18_finetuned_224px_augmented.pt"),
    ]:
        model = build_resnet18(pretrained=False).to(device)
        model.load_state_dict(torch.load(OUTPUTS_DIR / checkpoint, map_location=device))
        models[name] = (model, partial(preprocess_for_resnet, size=224))
    return models


def accuracy_on(model, preprocess, X, y, device):
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE)
    y_true, y_pred = collect_predictions(model, loader, device, preprocess)
    return (y_true == y_pred).float().mean().item()


def plot_robustness(rows, model_names, path):
    # One panel per corruption. Every line starts at the clean accuracy (severity 0).
    clean = {m: acc for c, _, m, acc in rows if c == "clean"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True)
    for ax, name in zip(axes.flat, SEVERITY):
        for model_name in model_names:
            by_level = {lvl: acc for c, lvl, m, acc in rows if c == name and m == model_name}
            ys = [clean[model_name], by_level[1], by_level[2], by_level[3]]
            ax.plot([0, 1, 2, 3], ys, marker="o", label=model_name)
        ax.set_title(name)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("severity (0 = clean)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("test accuracy")
    axes[1, 0].set_ylabel("test accuracy")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Accuracy under corruption (official test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    device = get_device()
    print(f"Using device: {device}")
    OUTPUTS_DIR.mkdir(exist_ok=True)

    X, y = to_tensors(load_test_split())
    print(f"Test images: {len(X)}")
    models = load_models(device)

    rows = []  # (corruption, severity, model name, accuracy)
    for model_name, (model, preprocess) in models.items():
        model.eval()
        clean = accuracy_on(model, preprocess, X, y, device)
        rows.append(("clean", 0, model_name, clean))
        for name in SEVERITY:
            for level in (1, 2, 3):
                Xc = corrupt(X, name, level)
                acc = accuracy_on(model, preprocess, Xc, y, device)
                rows.append((name, level, model_name, acc))
                print(f"{model_name:28s} {name:9s} severity {level}: {acc:.4f}")

    csv_path = OUTPUTS_DIR / "robustness_results.csv"
    with open(csv_path, "w") as f:
        f.write("corruption,severity,model,accuracy\n")
        for corruption, level, model_name, acc in rows:
            f.write(f"{corruption},{level},{model_name},{acc:.4f}\n")
    print(f"Saved results to {csv_path}")

    plot_path = OUTPUTS_DIR / "robustness_plot.png"
    plot_robustness(rows, list(models), plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
