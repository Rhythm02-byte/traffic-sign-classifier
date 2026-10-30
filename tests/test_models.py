import torch

from tsc.models import SmallCNN
from tsc.transfer import build_resnet18, freeze_backbone


def test_small_cnn_outputs_one_score_per_class():
    model = SmallCNN().eval()
    out = model(torch.zeros(4, 3, 32, 32))
    assert out.shape == (4, 43)


def test_resnet18_outputs_43_classes():
    model = build_resnet18(pretrained=False).eval()
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 43)


def test_freeze_backbone_leaves_only_the_head_trainable():
    model = build_resnet18(pretrained=False)
    freeze_backbone(model)

    trainable = sorted(name for name, p in model.named_parameters() if p.requires_grad)

    assert all(name.startswith("fc.") for name in trainable)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 512 * 43 + 43
