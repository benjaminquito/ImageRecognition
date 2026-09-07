#!/usr/bin/env python3
"""Train and evaluate the paper's MNIST convolutional network."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
from pathlib import Path

import numpy as np

from model import build_paper_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mnist"))
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def prepare_mnist(train_limit: int | None, test_limit: int | None):
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    if train_limit is not None:
        x_train, y_train = x_train[:train_limit], y_train[:train_limit]
    if test_limit is not None:
        x_test, y_test = x_test[:test_limit], y_test[:test_limit]

    x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
    x_test = x_test.astype("float32")[..., np.newaxis] / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def save_history(history: dict[str, list[float]], path: Path) -> None:
    columns = list(history)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", *columns])
        writer.writeheader()
        for epoch in range(len(history[columns[0]])):
            row = {name: history[name][epoch] for name in columns}
            writer.writerow({"epoch": epoch + 1, **row})


def save_curves(history: dict[str, list[float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, history["accuracy"], label="training")
    axes[0].plot(epochs, history["val_accuracy"], label="validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, history["loss"], label="training")
    axes[1].plot(epochs, history["val_loss"], label="validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Categorical cross-entropy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch size must be positive")

    set_reproducible_seed(args.seed)
    (x_train, y_train), (x_test, y_test) = prepare_mnist(
        args.train_limit, args.test_limit
    )
    model = build_paper_cnn(dropout=args.dropout)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = io.StringIO()
    model.summary(print_fn=lambda line: summary.write(line + "\n"))
    (output_dir / "model_summary.txt").write_text(summary.getvalue(), encoding="utf-8")

    trained = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(output_dir / "model.keras")
    save_history(trained.history, output_dir / "history.csv")
    save_curves(trained.history, output_dir / "training_curves.png")

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "seed": args.seed,
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "trainable_parameters": int(model.count_params()),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

