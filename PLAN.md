# PLAN.md — Implementation Plan

Derived from SPEC.md. Work through steps in order; each step has clear done criteria.

---

## Step 1 — Project Scaffold

**Goal:** Establish directory layout and dependencies.

Tasks:
- [ ] Create `requirements.txt` (torch, torchvision, pytest)
- [ ] Create `config.yaml` with all hard-coded values from SPEC.md extracted as keys
- [ ] Create empty files: `model.py`, `train.py`, `evaluate.py`, `tests/__init__.py`, `tests/test_model.py`

Done when: `pip install -r requirements.txt` succeeds; all files exist.

---

## Step 2 — Config Loader

**Goal:** Single place to read `config.yaml`; all other modules import from here.

Tasks:
- [ ] Create `config.py` with a `load_config(path)` function returning a dict
- [ ] Validate required keys are present; raise `ValueError` on missing keys

Done when: `python config.py` prints the loaded config without error.

---

## Step 3 — Data Loading Pipeline

**Goal:** Provide train and test `DataLoader` objects, MNIST normalized per spec.

Tasks:
- [ ] In `train.py`, implement `get_dataloaders(cfg)` using `torchvision.datasets.MNIST`
- [ ] Apply `transforms.Normalize(mean=0.1307, std=0.3081)`
- [ ] Use `batch_size` from config
- [ ] Return `(train_loader, test_loader)`

Done when: iterating `train_loader` yields batches of shape `(64, 1, 28, 28)` and `(64,)`.

---

## Step 4 — Model Definition

**Goal:** Dense network matching SPEC architecture.

Tasks:
- [ ] In `model.py`, implement `MNISTNet(cfg)` as `nn.Module`
- [ ] Layers: `Linear(784→256, ReLU)` → `Linear(256→128, ReLU)` → `Linear(128→10)`
- [ ] `forward()` flattens input from `(B,1,28,28)` to `(B,784)` before first layer
- [ ] No softmax in `forward()` (CrossEntropyLoss expects raw logits)
- [ ] Architecture must come from config, not hard-coded

Done when: `MNISTNet(cfg)(torch.zeros(4,1,28,28)).shape == (4,10)`.

---

## Step 5 — Training Pipeline

**Goal:** Train the model; print loss every 100 batches; save best checkpoint.

Tasks:
- [ ] In `train.py`, implement `train(cfg)` function
- [ ] Use `CrossEntropyLoss` and `Adam(lr=1e-3)` from config
- [ ] Loop for `epochs` (default 5) from config
- [ ] Print training loss every 100 batches
- [ ] After each epoch, compute and print validation/test accuracy
- [ ] Save model weights to `outputs/best_model.pt` when accuracy improves

Done when: running `python train.py` prints per-batch loss and per-epoch accuracy.

---

## Step 6 — Evaluation Pipeline

**Goal:** Load a checkpoint and report test accuracy.

Tasks:
- [ ] In `evaluate.py`, implement `evaluate(cfg)` function
- [ ] Load model weights from path in config
- [ ] Run inference over test set; compute and print accuracy
- [ ] Return accuracy value (for use in tests)

Done when: `python evaluate.py` prints test accuracy matching the training run.

---

## Step 7 — Tests

**Goal:** Automated checks for the acceptance criteria.

Tasks:
- [ ] In `tests/test_model.py`:
  - [ ] Test output shape: `forward(zeros(4,1,28,28))` → `(4,10)`
  - [ ] Test no NaN/Inf in output
  - [ ] Test `forward` is differentiable (loss.backward() does not raise)
- [ ] Bonus: smoke test that one training step reduces loss

Done when: `pytest tests/test_model.py` passes with no errors.

---

## Step 8 — Acceptance Criteria Verification

**Goal:** Confirm all spec criteria are met.

- [ ] Run `python train.py` for 5 epochs → test accuracy > 97%
- [ ] Confirm training loss is printed every 100 batches
- [ ] Confirm `pytest tests/test_model.py` passes
- [ ] Confirm all values (lr, epochs, batch_size, hidden sizes) come from `config.yaml`

Done when: all four bullets are checked off.

---

## Sequence Summary

```
Step 1 (scaffold) → Step 2 (config) → Step 3 (data) → Step 4 (model)
    → Step 5 (train) → Step 6 (evaluate) → Step 7 (tests) → Step 8 (verify)
```

Each step depends on the previous; implement in order.
