import torch

from tsc.train import normalize
from tsc.transfer import IMAGENET_MEAN, IMAGENET_STD, preprocess_for_resnet


def test_normalize_maps_0_to_minus_one_and_255_to_one():
    x = torch.tensor([[[[0.0, 255.0]]]]).to(torch.uint8)
    out = normalize(x)
    assert torch.allclose(out.flatten(), torch.tensor([-1.0, 1.0]))


def test_resnet_preprocess_resizes_and_uses_imagenet_statistics():
    x = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)

    out = preprocess_for_resnet(x, size=224)

    assert out.shape == (2, 3, 224, 224)
    expected = (0 - IMAGENET_MEAN) / IMAGENET_STD  # black pixels, normalised per channel
    assert torch.allclose(out[0, :, 0, 0], expected.flatten(), atol=1e-6)
