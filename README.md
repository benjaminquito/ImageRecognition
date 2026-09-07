# LiDAR and Camera Image Recognition

Reproducible code companion for:

> Quito, B. and Esmahi, L. (2023). “Compare and Contrast LiDAR and Non-LiDAR
> Technology in an Autonomous Vehicle: Developing a Safety Framework.”
> *Open Journal of Safety Science and Technology*, 13, 101–131.
> [https://doi.org/10.4236/ojsst.2023.133006](https://doi.org/10.4236/ojsst.2023.133006)

The project reconstructs the executable parts of the paper and separates them
from the proposed (not yet completed) vehicle experiment.

## What is reproduced

| Paper component | Script | Output |
|---|---|---|
| 28 × 28 CNN on MNIST | `train_mnist.py` | Saved model, metrics, history, and accuracy/loss plots |
| ImageNet VGG16 inference | `predict_vgg16.py` | Top-k labels and probabilities for an input image |
| LiDAR/camera safety framework | `safety_framework.py` | Blank 25-run protocol and aggregated weather results |

The MNIST network follows the architecture in sections 3.6.1–3.6.4: two
5 × 5 convolution layers (32 and 64 filters), max pooling, a 1,024-unit dense
layer, and a 10-class output. It contains 3,274,634 trainable parameters when
dropout is disabled, matching the paper.

The VGG16 script follows sections 3.7–3.8: resize to 224 × 224, apply Keras'
VGG16 preprocessing, use ImageNet weights, and decode the ten most likely
classes.

## Important interpretation

The paper reports several different numbers:

- **99.29%** is the reported MNIST evaluation accuracy after 25 epochs.
- **94.63%** is the top prediction probability for one bee photograph using
  pretrained VGG16. It is a confidence score for one sample, not a dataset
  accuracy and not a comparison of LiDAR against camera data.
- The LiDAR/camera weather tables are an experimental proposal. No completed
  sensor dataset or table values are published in the paper.

Accordingly, this repository does not claim to reproduce a measured 94.63%
LiDAR-vs-camera result. Exact floating-point results can vary by platform,
TensorFlow version, initialization, and the input image.

## Setup

The paper used Python 3.10.4 and TensorFlow. A clean Python 3.10 environment is
recommended.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the MNIST experiment

Full 25-epoch reproduction:

```bash
python train_mnist.py --epochs 25 --output-dir artifacts/mnist
```

Optional dropout enhancement described in section 3.6.5:

```bash
python train_mnist.py --epochs 25 --dropout 0.5 \
  --output-dir artifacts/mnist-dropout
```

For a quick pipeline check:

```bash
python train_mnist.py --epochs 1 --train-limit 2048 --test-limit 512 \
  --output-dir artifacts/smoke
```

Each run writes `model.keras`, `metrics.json`, `history.csv`,
`training_curves.png`, and `model_summary.txt`.

## Run VGG16 inference

```bash
python predict_vgg16.py path/to/image.jpg --top 10 \
  --output artifacts/vgg16-prediction.json
```

The ImageNet weights download automatically on first use. To revisit the bee
example, supply a bee image whose reuse rights you have; the original image is
not distributed with the article.

## Use the safety-framework scaffold

Create a protocol with 25 runs for each combination of driver mode,
technology, and weather condition:

```bash
python safety_framework.py init data/safety_runs.csv --runs 25
```

Fill the measurement columns in the CSV, then aggregate results:

```bash
python safety_framework.py summarize data/safety_runs.csv \
  --output artifacts/safety_summary.csv
```

The long-form protocol captures accuracy, confidence, latency, obstacle
detection, stopping distance, and notes. These fields support the paper's
proposed extensions while keeping missing measurements blank.

The prose names six conditions: sunny (`S`), cloudy (`C`), daytime rain
(`DR`), fog (`F`), nighttime rain (`NR`), and snow (`SW`). Tables 3–6 also
contain an unexplained `R` column. The scaffold preserves `R` as `rain` so the
published table layout can be represented without silently discarding it.

## Tests

The lightweight tests do not download datasets or TensorFlow weights:

```bash
python -m unittest -v
python -m compileall -q model.py train_mnist.py predict_vgg16.py \
  safety_framework.py
```

## Reproducibility boundary

This code reproduces the published software procedures as closely as the paper
allows. A direct LiDAR-versus-camera safety comparison still requires paired,
time-synchronized sensor captures, ground-truth labels, weather metadata, and
the 25 physical runs per condition proposed in section 4. Those data are not
included in the publication.
