Goal
  Fully-connected neural network for MNIST digit classification in PyTorch.

Architecture
  Input(784) → Linear(256, ReLU) → Linear(128, ReLU) → Output(10)

Constraints
  - PyTorch only; no conv layers; no external model libraries (e.g. timm)
  - Python 3.11, type hints required, no raw loops over batches

Data
  torchvision MNIST · normalize: mean=0.1307, std=0.3081
  batch_size=64

Training
  Loss:      CrossEntropyLoss
  Optimizer: Adam, lr=1e-3
  Epochs:    5

Acceptance Criteria
  - Test accuracy > 97% by epoch 5
  - Training loss printed every 100 batches
  - pytest tests/test_model.py passes (input/output shapes, forward pass)

Output Files
  model.py   train.py   evaluate.py   tests/test_model.py