# MNIST Digit Classifier

A fully-connected neural network trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit dataset. Built with PyTorch using a spec-driven development process — every pipeline behaviour is controlled through a single config file.

**Achieves 97.7% test accuracy in 5 epochs on CPU.**

---

## How it works

The model is a simple dense (fully-connected) network:

```
Input (784) → Linear → ReLU → Linear → ReLU → Output (10)
```

It takes a 28×28 grayscale image, flattens it to 784 numbers, passes it through two hidden layers, and outputs a score for each digit 0–9. The digit with the highest score is the prediction.

---

## Project structure

```
mnist-sdd/
├── config.yaml       ← the one file you need to edit to change anything
├── config.py         ← loads and validates config.yaml
├── model.py          ← neural network definition
├── train.py          ← data loading + training pipeline
├── evaluate.py       ← loads a checkpoint and reports test accuracy
├── tests/
│   └── test_model.py ← pytest tests
└── requirements.txt
```

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/ksrinivas1225/mnist-sdd.git
cd mnist-sdd
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

This installs PyTorch, torchvision, PyYAML, and pytest.

---

## Quickstart

### Train the model

```bash
python train.py
```

On the first run, MNIST is downloaded automatically to `./data/`. Training prints the loss every 100 batches and the test accuracy at the end of each epoch. The best checkpoint is saved to `./outputs/best_model.pt`.

Example output:

```
epoch 1 | batch  100 | loss 0.6275
epoch 1 | batch  200 | loss 0.3122
...
epoch 1 complete | test accuracy: 96.61%
  -> checkpoint saved (best acc: 96.61%)
...
epoch 5 complete | test accuracy: 97.72%
  -> checkpoint saved (best acc: 97.72%)
```

### Evaluate a saved checkpoint

```bash
python evaluate.py
```

Loads the checkpoint specified in `config.yaml` and prints the test accuracy:

```
test accuracy: 97.72%  (9772/10000)
```

### Run the tests

```bash
pytest tests/ -v
```

```
tests/test_model.py::test_output_shape          PASSED
tests/test_model.py::test_no_nan_or_inf         PASSED
tests/test_model.py::test_backward              PASSED
tests/test_model.py::test_one_step_reduces_loss PASSED
```

---

## Configuration

Everything is controlled through `config.yaml`. You never need to edit the source files.

```yaml
seed: 42                        # random seed for reproducibility

data:
  root: ./data                  # where MNIST is downloaded
  batch_size: 64                # images per training batch
  num_workers: 2                # parallel data loading workers
  normalize:
    mean: 0.1307                # MNIST dataset mean (do not change)
    std: 0.3081                 # MNIST dataset std  (do not change)

model:
  hidden_layers: [256, 128]     # units in each hidden layer
  activation: relu              # relu | tanh | sigmoid

train:
  epochs: 5                     # number of training epochs
  optimizer: adam               # adam | sgd
  lr: 1.0e-3                    # learning rate
  checkpoint_path: ./outputs/best_model.pt

evaluate:
  checkpoint_path: ./outputs/best_model.pt
  device: cpu                   # cpu | cuda
```

### Common customisations

**Train for longer**
```yaml
train:
  epochs: 20
```

**Use a deeper network**
```yaml
model:
  hidden_layers: [512, 256, 128]
```

**Switch to SGD**
```yaml
train:
  optimizer: sgd
  lr: 0.1
```

**Run on GPU** (if available)
```yaml
evaluate:
  device: cuda
```

---

## How the code is organised

### `config.py`

Loads `config.yaml` and validates that all required keys are present. Every other module calls `load_config()` at startup.

```python
from config import load_config
cfg = load_config()          # reads config.yaml
cfg["train"]["lr"]           # 0.001
```

### `model.py`

Defines `MNISTNet`, which builds its layers from `config.yaml` at construction time.

```python
from model import MNISTNet
model = MNISTNet(cfg)        # architecture from config
logits = model(images)       # shape (batch, 10) — raw scores, no softmax
```

### `train.py`

Contains two functions:
- `get_dataloaders(cfg)` — returns `(train_loader, test_loader)`
- `train(cfg)` — full training loop with checkpointing

### `evaluate.py`

Contains `evaluate(cfg)` — loads a checkpoint and returns test accuracy as a float.

---

## Spec-driven development

This project was built following a spec-driven process:

- `SPEC.md` defines what the code must do (requirements + acceptance criteria)
- `PLAN.md` breaks the spec into 8 ordered implementation steps
- `CLAUDE.md` guides the AI assistant (Claude Code) on conventions

Code was written only after the spec was agreed upon, and each step was verified against its acceptance criteria before moving to the next.
