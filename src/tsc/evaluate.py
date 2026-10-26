"""Step 3: measure the baseline CNN on the official GTSRB test set."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from tsc.data import load_test_split, to_tensors
from tsc.models import SmallCNN
from tsc.train import BATCH_SIZE, get_device, normalize

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"
CHECKPOINT = OUTPUTS_DIR / "baseline.pt"
NUM_CLASSES = 43


def collect_predictions(model, loader, device, preprocess):
    # Run the model over the whole test set and keep the true and predicted class of every image
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(preprocess(xb).to(device)).argmax(dim=1).cpu()
            true_labels.append(yb)
            pred_labels.append(preds)
    return torch.cat(true_labels), torch.cat(pred_labels)


def per_class_accuracy(y_true, y_pred):
    # For each class: how many test images it has, how many were right, and the share right
    rows = []
    for class_id in range(NUM_CLASSES):
        mask = y_true == class_id
        total = mask.sum().item()
        correct = (y_pred[mask] == class_id).sum().item()
        rows.append((class_id, total, correct, correct / total))
    return rows


def save_per_class_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "num_images", "correct", "accuracy"])
        for class_id, total, correct, acc in rows:
            writer.writerow([class_id, total, correct, f"{acc:.4f}"])


def confusion_matrix(y_true, y_pred):
    # Entry [t, p] counts test images of true class t that the model predicted as class p.
    # Each (t, p) pair becomes one number, t * 43 + p, and bincount counts how often each occurs.
    pairs = y_true * NUM_CLASSES + y_pred
    counts = torch.bincount(pairs, minlength=NUM_CLASSES * NUM_CLASSES)
    return counts.reshape(NUM_CLASSES, NUM_CLASSES)


def plot_confusion_matrix(cm, path, title):
    # Divide each row by its total, so every true class reads as a share of its own images
    row_totals = cm.sum(dim=1, keepdim=True).clamp(min=1)
    shares = (cm / row_totals).numpy()

    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(shares, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_model(model, preprocess, device, csv_path, cm_path, title):
    # Test images are loaded the same way for every model, so the results are comparable
    X_test, y_test = to_tensors(load_test_split())
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

    y_true, y_pred = collect_predictions(model, test_loader, device, preprocess)
    overall = (y_true == y_pred).float().mean().item()
    print(f"Test images: {len(y_true)}")
    print(f"Test accuracy: {overall:.4f}")

    rows = per_class_accuracy(y_true, y_pred)
    save_per_class_csv(rows, csv_path)
    print("Five weakest classes (class id: accuracy):")
    for class_id, total, correct, acc in sorted(rows, key=lambda r: r[3])[:5]:
        print(f"  class {class_id:2d}: {acc:.3f} ({correct}/{total})")
    print(f"Saved per-class accuracy to {csv_path}")

    plot_confusion_matrix(confusion_matrix(y_true, y_pred), cm_path, title)
    print(f"Saved confusion matrix to {cm_path}")
    return overall


def main():
    device = get_device()
    print(f"Using device: {device}")
    OUTPUTS_DIR.mkdir(exist_ok=True)

    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    print(f"Loaded checkpoint: {CHECKPOINT}")

    # Same resize and same normalisation as training (normalize() comes from train.py)
    evaluate_model(
        model,
        normalize,
        device,
        csv_path=OUTPUTS_DIR / "per_class_accuracy.csv",
        cm_path=OUTPUTS_DIR / "confusion_matrix.png",
        title="Baseline CNN confusion matrix (each row sums to 1)",
    )


if __name__ == "__main__":
    main()
