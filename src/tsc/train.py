"""Step 2: train the baseline CNN from scratch."""
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tsc.data import load_train_split, split_train_val, to_tensors
from tsc.models import SmallCNN

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"
SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    # Apple GPU if available, otherwise CPU
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def normalize(x):
    # Pixels are 0..255. Scale them to -1..1, which helps training converge.
    return x.float() / 127.5 - 1.0


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()  # turns on dropout and BatchNorm training behaviour
    for xb, yb in loader:
        xb, yb = normalize(xb).to(device), yb.to(device)
        optimizer.zero_grad()  # clear gradients from the previous step
        loss = loss_fn(model(xb), yb)
        loss.backward()  # compute gradients
        optimizer.step()  # update weights


def accuracy(model, loader, device):
    model.eval()  # turns dropout off and uses BatchNorm's learned statistics
    correct, total = 0, 0
    with torch.no_grad():  # no gradients needed for evaluation
        for xb, yb in loader:
            preds = model(normalize(xb).to(device)).argmax(dim=1).cpu()
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / total


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    X, y = to_tensors(load_train_split())
    (X_train, y_train), (X_val, y_val) = split_train_val(X, y, seed=SEED)
    print(f"Train: {len(X_train)} images, validation: {len(X_val)} images")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    checkpoint = OUTPUTS_DIR / "baseline.pt"
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_acc = accuracy(model, val_loader, device)
        print(f"epoch {epoch:2d}  validation accuracy {val_acc:.3f}")
        # Keep the best model on validation, not the last one
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint)

    print(f"Best validation accuracy: {best_acc:.3f}, saved to {checkpoint}")


if __name__ == "__main__":
    main()
