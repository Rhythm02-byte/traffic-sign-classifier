"""Classify your own traffic-sign photos with a trained model.

The model was trained on tightly cropped signs, so crop each photo to the sign before running it.
Full street scenes are not handled: the model won't find the sign in them.

Example:
    PYTHONPATH=src python -m tsc.predict photo1.jpg photo2.png --model resnet-augmented
"""
import argparse
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from tsc.data import IMG_SIZE
from tsc.models import SmallCNN
from tsc.train import get_device, normalize
from tsc.transfer import build_resnet18, preprocess_for_resnet

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"
CHECKPOINTS = {
    "baseline": "baseline.pt",
    "resnet": "resnet18_finetuned_224px.pt",
    "resnet-augmented": "resnet18_finetuned_224px_augmented.pt",
}
# Same resize as the training data (see to_tensors in data.py)
to_training_size = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])


def load_model(name, device):
    checkpoint = OUTPUTS_DIR / CHECKPOINTS[name]
    if name == "baseline":
        model, preprocess = SmallCNN(), normalize
    else:
        model, preprocess = build_resnet18(pretrained=False), partial(preprocess_for_resnet, size=224)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval(), preprocess


def predict_image(model, preprocess, path, device, top_k=3):
    return predict_pil(model, preprocess, Image.open(path), device, top_k)


def predict_pil(model, preprocess, image, device, top_k=3):
    x = to_training_size(image.convert("RGB")).unsqueeze(0)  # shape (1, 3, 32, 32), uint8
    with torch.no_grad():
        probs = torch.softmax(model(preprocess(x).to(device)), dim=1)[0].cpu()
    top = torch.topk(probs, top_k)
    return [(int(class_id), float(p)) for p, class_id in zip(top.values, top.indices)]


def main():
    parser = argparse.ArgumentParser(description="Classify traffic-sign photos (cropped to the sign).")
    parser.add_argument("images", nargs="+", type=Path, help="one or more image files")
    parser.add_argument("--model", choices=list(CHECKPOINTS), default="resnet-augmented")
    parser.add_argument("--top", type=int, default=3, help="how many guesses to show per image")
    args = parser.parse_args()

    device = get_device()
    model, preprocess = load_model(args.model, device)
    print(f"Model: {args.model} ({CHECKPOINTS[args.model]}), device: {device}")
    for path in args.images:
        print(path)
        for class_id, p in predict_image(model, preprocess, path, device, args.top):
            print(f"  class {class_id:2d}: {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
