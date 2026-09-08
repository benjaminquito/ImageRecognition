"""Neural-network definitions reconstructed from the paper."""

from __future__ import annotations


def build_paper_cnn(dropout: float = 0.0):
    """Build the 28x28 CNN described in sections 3.6.1-3.6.4.

    TensorFlow is imported lazily so metadata and safety utilities remain
    usable without initializing the ML runtime.
    """
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the interval [0, 1)")

    from tensorflow import keras

    layers = [
        keras.layers.Input(shape=(28, 28, 1), name="image"),
        keras.layers.Conv2D(
            32, kernel_size=(5, 5), padding="same", activation="relu"
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(
            64, kernel_size=(5, 5), padding="same", activation="relu"
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation="relu"),
    ]
    if dropout:
        layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(10, activation="softmax"))

    model = keras.Sequential(layers, name="paper_mnist_cnn")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

