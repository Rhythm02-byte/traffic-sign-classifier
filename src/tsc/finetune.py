"""Fine-tuning: unfreeze the whole ResNet-18 and train it at a low learning rate."""
import argparse
import math
from functools import partial

import torch
from torch import nn
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, TensorDataset

from tsc.data import load_train_split, split_train_val, to_tensors
from tsc.evaluate import evaluate_model
from tsc.train import BATCH_SIZE, SEED, get_device, set_seed
from tsc.transfer import (
    INPUT_SIZE,
    OUTPUTS_DIR,
    build_resnet18,
    preprocess_for_resnet,
    size_tag,
    validation_accuracy,
)

EPOCHS = 5
# Ten times smaller than the head's learning rate. The pretrained layers should adjust slowly, not be overwritten.
LR = 1e-4


def augment_batch(xb):
    # Training-only augmentation. Each batch is left clean, blurred, or given noise, chosen at random.
    # The ranges are narrower than the robustness test's strengths, so the test still includes unseen corruptions.
    x = xb.float()
    choice = torch.randint(0, 3, (1,)).item()
    if choice == 1:
        sigma = torch.empty(1).uniform_(0.5, 1.5).item()
        k = 2 * math.ceil(3 * sigma) + 1
        x = TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])
    elif choice == 2:
        std = torch.empty(1).uniform_(5, 25).item()
        x = x + torch.randn_like(x) * std
    return x.clamp(0, 255).round().to(torch.uint8)


def train_all_layers(model, loader, optimizer, loss_fn, device, preprocess, augment=False):
    model.train()  # every layer learns now, including BatchNorm
    for xb, yb in loader:
        if augment:
            xb = augment_batch(xb)
        xb, yb = preprocess(xb).to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()


def main(size=INPUT_SIZE, seed=SEED, augment=False):
    set_seed(seed)  # controls batch order and dropout; the train/validation split stays at SEED
    device = get_device()
    print(f"Using device: {device}, input size {size}px, seed {seed}")
    OUTPUTS_DIR.mkdir(exist_ok=True)
    preprocess = partial(preprocess_for_resnet, size=size)
    frozen_checkpoint = OUTPUTS_DIR / f"resnet18_frozen{size_tag(size)}.pt"
    # The default seed keeps the original file names. Other seeds get a suffix.
    run_tag = size_tag(size) + ("" if seed == SEED else f"_seed{seed}") + ("_augmented" if augment else "")
    checkpoint = OUTPUTS_DIR / f"resnet18_finetuned{run_tag}.pt"

    # Same data and split as the baseline and the frozen model
    X, y = to_tensors(load_train_split())
    (X_train, y_train), (X_val, y_val) = split_train_val(X, y, seed=SEED)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    # Start from the frozen model at the same input size, whose new head is already trained
    model = build_resnet18(pretrained=False).to(device)
    model.load_state_dict(torch.load(frozen_checkpoint, map_location=device))
    print(f"Starting from frozen checkpoint: {frozen_checkpoint}")
    print(f"Starting validation accuracy: {validation_accuracy(model, val_loader, device, preprocess):.3f}")

    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {total:,} of {total:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_all_layers(model, train_loader, optimizer, loss_fn, device, preprocess, augment)
        val_acc = validation_accuracy(model, val_loader, device, preprocess)
        print(f"epoch {epoch:2d}  validation accuracy {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint)

    print(f"Best validation accuracy: {best_acc:.3f}, saved to {checkpoint}")

    # Evaluate the best fine-tuned checkpoint on the official test set
    model = build_resnet18(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"Loaded checkpoint: {checkpoint}")
    evaluate_model(
        model,
        preprocess,
        device,
        csv_path=OUTPUTS_DIR / f"resnet18_finetuned{run_tag}_per_class_accuracy.csv",
        cm_path=OUTPUTS_DIR / f"resnet18_finetuned{run_tag}_confusion_matrix.png",
        title=f"ResNet-18, fine-tuned, {size}px, seed {seed}{', augmented' if augment else ''} (each row sums to 1)",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=INPUT_SIZE, help="input resolution in pixels")
    parser.add_argument("--seed", type=int, default=SEED, help="seed for batch order and dropout")
    parser.add_argument("--augment", action="store_true", help="train with blur and noise augmentation")
    args = parser.parse_args()
    main(size=args.size, seed=args.seed, augment=args.augment)
