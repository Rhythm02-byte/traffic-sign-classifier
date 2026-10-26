"""Transfer learning: pretrained ResNet-18 with a new 43-class head, backbone frozen."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from tsc.data import load_train_split, split_train_val, to_tensors
from tsc.evaluate import collect_predictions, evaluate_model
from tsc.train import BATCH_SIZE, SEED, get_device, set_seed

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"
CHECKPOINT = OUTPUTS_DIR / "resnet18_frozen.pt"
INPUT_SIZE = 128  # ResNet was pretrained on 224 px; 128 keeps training fast on a laptop
EPOCHS = 5
LR = 1e-3

# The mean and standard deviation that ImageNet pretraining used.
# The pretrained weights expect inputs scaled this way.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess_for_resnet(x):
    # x holds uint8 images of shape (N, 3, 32, 32), as stored by to_tensors
    x = x.float() / 255.0  # 0..255 -> 0..1
    x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def build_resnet18(pretrained=True, num_classes=43):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    # The original head has 1000 ImageNet classes. Replace it with one for our 43 signs.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model):
    # Only the new head (fc) keeps learning. Everything else keeps its pretrained values.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def set_train_mode(model):
    model.train()
    # BatchNorm in the frozen backbone stays in eval mode, so its running statistics
    # keep the values learned on ImageNet instead of drifting toward GTSRB
    for name, module in model.named_children():
        if name != "fc":
            module.eval()


def train_head(model, loader, optimizer, loss_fn, device):
    set_train_mode(model)
    for xb, yb in loader:
        xb, yb = preprocess_for_resnet(xb).to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()


def validation_accuracy(model, loader, device):
    y_true, y_pred = collect_predictions(model, loader, device, preprocess_for_resnet)
    return (y_true == y_pred).float().mean().item()


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Same data and same split as the baseline (same seed). Only the model and preprocessing change.
    X, y = to_tensors(load_train_split())
    (X_train, y_train), (X_val, y_val) = split_train_val(X, y, seed=SEED)
    print(f"Train: {len(X_train)} images, validation: {len(X_val)} images")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = build_resnet18(pretrained=True).to(device)
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} of {total:,}")

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_head(model, train_loader, optimizer, loss_fn, device)
        val_acc = validation_accuracy(model, val_loader, device)
        print(f"epoch {epoch:2d}  validation accuracy {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)

    print(f"Best validation accuracy: {best_acc:.3f}, saved to {CHECKPOINT}")

    # Evaluate the saved best checkpoint on the official test set
    model = build_resnet18(pretrained=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    print(f"Loaded checkpoint: {CHECKPOINT}")
    evaluate_model(
        model,
        preprocess_for_resnet,
        device,
        csv_path=OUTPUTS_DIR / "resnet18_frozen_per_class_accuracy.csv",
        cm_path=OUTPUTS_DIR / "resnet18_frozen_confusion_matrix.png",
        title="ResNet-18, frozen backbone (each row sums to 1)",
    )


if __name__ == "__main__":
    main()
