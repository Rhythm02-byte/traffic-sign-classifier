import torch
from PIL import Image

from tsc.data import IMG_SIZE, split_train_val, to_tensors


def test_to_tensors_resizes_and_keeps_labels():
    samples = [(Image.new("RGB", (50, 40)), 3), (Image.new("RGB", (20, 60)), 7)]

    X, y = to_tensors(samples)

    assert X.shape == (2, 3, IMG_SIZE, IMG_SIZE)
    assert X.dtype == torch.uint8
    assert y.tolist() == [3, 7]


def test_split_is_disjoint_and_covers_everything():
    X = torch.zeros(100, 3, IMG_SIZE, IMG_SIZE, dtype=torch.uint8)
    y = torch.arange(100)  # unique label per image, so we can see which image lands where

    (X_train, y_train), (X_val, y_val) = split_train_val(X, y, val_fraction=0.1, seed=42)

    assert len(y_val) == 10
    assert len(y_train) == 90
    assert set(y_train.tolist()).isdisjoint(y_val.tolist())
    assert sorted(y_train.tolist() + y_val.tolist()) == list(range(100))


def test_split_is_reproducible_with_the_same_seed():
    X = torch.zeros(50, 3, IMG_SIZE, IMG_SIZE, dtype=torch.uint8)
    y = torch.arange(50)

    _, (_, val_a) = split_train_val(X, y, seed=1)
    _, (_, val_b) = split_train_val(X, y, seed=1)

    assert torch.equal(val_a, val_b)
