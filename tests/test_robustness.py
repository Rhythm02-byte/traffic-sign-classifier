import torch

from tsc.robustness import SEVERITY, corrupt


def random_images(n=6):
    torch.manual_seed(0)
    return torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8)


def test_every_corruption_keeps_shape_and_dtype():
    X = random_images()
    for name in SEVERITY:
        for level in (1, 2, 3):
            out = corrupt(X, name, level)
            assert out.shape == X.shape, name
            assert out.dtype == torch.uint8, name


def test_darker_lowers_brightness_and_brighter_raises_it():
    X = random_images()
    assert corrupt(X, "darker", 3).float().mean() < X.float().mean()
    assert corrupt(X, "brighter", 1).float().mean() > X.float().mean()


def test_noise_is_the_same_every_run():
    X = random_images()
    assert torch.equal(corrupt(X, "noise", 2), corrupt(X, "noise", 2))
