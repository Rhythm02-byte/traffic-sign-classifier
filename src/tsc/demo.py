"""Gradio demo: upload a cropped traffic-sign photo and see the model's top guesses.

Run from the repo root with the environment active:
    PYTHONPATH=src python -m tsc.demo
Then open the local address it prints (http://127.0.0.1:7860 by default).
"""
import gradio as gr

from tsc.predict import CHECKPOINTS, load_model, predict_pil
from tsc.train import get_device

DEFAULT_MODEL = "resnet-augmented"
DESCRIPTION = (
    "Upload a photo of a German traffic sign, cropped tightly around the sign. "
    "The model was trained on tightly cropped signs, so full street scenes will give unreliable results."
)


def build_demo(device):
    # Load every model once at start-up, so switching models in the dropdown is instant
    models = {name: load_model(name, device) for name in CHECKPOINTS}

    def classify(image, model_name):
        if image is None:
            return None
        model, preprocess = models[model_name]
        top = predict_pil(model, preprocess, image, device, top_k=3)
        return {f"class {class_id}": p for class_id, p in top}

    return gr.Interface(
        fn=classify,
        inputs=[
            gr.Image(type="pil", label="Traffic sign photo"),
            gr.Dropdown(choices=list(CHECKPOINTS), value=DEFAULT_MODEL, label="Model"),
        ],
        outputs=gr.Label(num_top_classes=3, label="Top guesses (GTSRB class IDs)"),
        title="Traffic Sign Classifier (GTSRB)",
        description=DESCRIPTION,
    )


if __name__ == "__main__":
    device = get_device()
    print(f"Loading models on {device}")
    build_demo(device).launch()
