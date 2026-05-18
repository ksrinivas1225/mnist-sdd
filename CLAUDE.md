# MNIST Handwritten Digit Recognition — Claude Code Guide

## Project Overview

Spec-driven development (SDD) project: a dense neural network in PyTorch that classifies 28×28 grayscale MNIST images into digits 0–9. All behavior is driven by `config.yaml`.

## Repository Layout

```
mnist-sdd/
├── CLAUDE.md           # This file — project guide for Claude Code
├── SPEC.md             # Functional specification (source of truth)
├── config.yaml         # Master config — controls all pipelines
├── src/
│   ├── data.py         # Data loading pipeline
│   ├── model.py        # Dense network definition
│   ├── train.py        # Training + validation pipeline
│   ├── infer.py        # Inference pipeline
│   └── tune.py         # Hyperparameter tuning pipeline
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_train.py
│   └── test_infer.py
├── outputs/            # Checkpoints, logs, tuning results (git-ignored)
└── requirements.txt
```

## Spec-Driven Development Process

1. **Spec first** — all features must be described in `SPEC.md` before implementation begins.
2. **Config-driven** — no pipeline behavior is hard-coded; every tunable parameter lives in `config.yaml`.
3. **Implement to spec** — code must satisfy the acceptance criteria in `SPEC.md`. Do not add features not in the spec.
4. **Test against spec** — each acceptance criterion maps to at least one test.

## Tech Stack

- Python 3.10+
- PyTorch (training, model, inference)
- torchvision (MNIST dataset download and transforms)
- PyYAML (config loading)
- Optuna (hyperparameter tuning)
- pytest (testing)

## Config File (`config.yaml`)

All pipelines read from `config.yaml`. Keys are namespaced by pipeline:

```yaml
data:    { ... }
model:   { ... }
train:   { ... }
infer:   { ... }
tune:    { ... }
```

Never hard-code values that belong in the config.

## Coding Conventions

- One public function or class per module file where possible.
- Type hints on all function signatures.
- No docstrings unless the logic is genuinely non-obvious; prefer clear names.
- Raise `ValueError` for invalid config values at startup, not deep in the call stack.
- Log to stdout using Python's `logging` module; log level controlled by config.

## Running Pipelines

```bash
python src/train.py          # train + validate
python src/infer.py          # run inference on a sample
python src/tune.py           # hyperparameter search
```

## Testing

```bash
pytest tests/
```

All tests must pass before a feature is considered complete.
