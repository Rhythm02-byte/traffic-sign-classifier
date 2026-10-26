"""Step 1c: download GTSRB and look at it."""
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
IMG_SIZE = 32


def load_train_split():
    # download=True fetches the dataset once; later runs reuse the files in data/
    return datasets.GTSRB(root=DATA_DIR, split="train", download=True)


def load_test_split():
    # download=False: the test files were prepared by hand, we never fetch them over the network
    return datasets.GTSRB(root=DATA_DIR, split="test", download=False)


def plot_samples(dataset, path, n=12, seed=0):
    # a fixed seed picks the same random images every run
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n)

    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for ax, idx in zip(axes.flat, indices):
        image, label = dataset[idx]
        ax.imshow(image)
        ax.set_title(f"class {label}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_class_distribution(dataset, path):
    counts = Counter(label for _, label in dataset)
    classes = sorted(counts)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(classes, [counts[c] for c in classes])
    ax.set_xlabel("class id")
    ax.set_ylabel("number of training images")
    ax.set_title("GTSRB training set: images per class")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    return counts


def to_tensors(dataset):
    # Resize every image once and stack them, so training does not decode the PPM files every epoch
    resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])
    images, labels = [], []
    for image, label in dataset:
        images.append(resize(image))
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def split_train_val(X, y, val_fraction=0.1, seed=42):
    # Shuffle once with a fixed seed, then hold out the first 10% as validation
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=generator)
    n_val = int(len(X) * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])


if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(exist_ok=True)
    train = load_train_split()
    print(f"Training images: {len(train)}")

    samples_path = OUTPUTS_DIR / "samples.png"
    plot_samples(train, samples_path)
    print(f"Saved sample grid to {samples_path}")

    dist_path = OUTPUTS_DIR / "class_distribution.png"
    counts = plot_class_distribution(train, dist_path)
    print(f"Saved class distribution to {dist_path}")
    print(f"Classes: {len(counts)}")
    print(f"Fewest images in one class: {min(counts.values())}")
    print(f"Most images in one class:   {max(counts.values())}")
