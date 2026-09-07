#!/usr/bin/env python3
"""Classify one image with ImageNet-pretrained VGG16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def predict(image_path: Path, top: int = 10) -> list[dict[str, object]]:
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
    batch = preprocess_input(batch)
    probabilities = VGG16(weights="imagenet").predict(batch, verbose=0)
    decoded = decode_predictions(probabilities, top=top)[0]
    return [
        {"synset": synset, "label": label, "probability": float(probability)}
        for synset, label, probability in decoded
    ]


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()

