"""ImageNet-pretrained VGG16 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def predict(image_path: Path, top: int = 10) -> list[dict[str, object]]:
    """Return the top ImageNet predictions for one image."""
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not 1 <= top <= 1000:
        raise ValueError("top must be between 1 and 1000")

    from tensorflow.keras.applications.vgg16 import (
        VGG16,
        decode_predictions,
        preprocess_input,
    )
    from tensorflow.keras.utils import img_to_array, load_img

    image = load_img(image_path, target_size=(224, 224))
    batch = np.expand_dims(img_to_array(image), axis=0)
    probabilities = VGG16(weights="imagenet").predict(
        preprocess_input(batch), verbose=0
    )
    decoded = decode_predictions(probabilities, top=top)[0]
    return [
        {"synset": synset, "label": label, "probability": float(probability)}
        for synset, label, probability in decoded
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = {
        "image": str(args.image),
        "model": "VGG16",
        "weights": "ImageNet",
        "predictions": predict(args.image, args.top),
    }
    rendered = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

