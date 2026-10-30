from functools import partial

from PIL import Image

from tsc.predict import predict_image
from tsc.transfer import build_resnet18, preprocess_for_resnet


def test_predict_returns_top_k_valid_class_ids(tmp_path):
    photo = tmp_path / "sign.png"
    Image.new("RGB", (60, 60), color=(200, 30, 30)).save(photo)
    model = build_resnet18(pretrained=False).eval()  # untrained, but the output shape is what we check
    preprocess = partial(preprocess_for_resnet, size=64)

    results = predict_image(model, preprocess, photo, device="cpu", top_k=3)

    assert len(results) == 3
    assert all(0 <= class_id < 43 for class_id, _ in results)
    assert all(0.0 <= p <= 1.0 for _, p in results)
    probs = [p for _, p in results]
    assert probs == sorted(probs, reverse=True)
