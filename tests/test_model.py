import torch
import torch.nn as nn
import pytest

from config import load_config
from model import MNISTNet

@pytest.fixture(scope="module")
def cfg():
    return load_config()

@pytest.fixture(scope="module")
def model(cfg):
    return MNISTNet(cfg)

@pytest.fixture(scope="module")
def dummy():
    return torch.zeros(4, 1, 28, 28)


def test_output_shape(model, dummy):
    out = model(dummy)
    assert out.shape == (4, 10)

def test_no_nan_or_inf(model, dummy):
    out = model(dummy)
    assert torch.isfinite(out).all()

def test_backward(model, dummy):
    out = model(dummy)
    labels = torch.zeros(4, dtype=torch.long)
    loss = nn.CrossEntropyLoss()(out, labels)
    loss.backward()

def test_one_step_reduces_loss(cfg):
    model = MNISTNet(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    criterion = nn.CrossEntropyLoss()
    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))

    model.train()
    loss_before = criterion(model(images), labels).item()
    optimizer.zero_grad()
    criterion(model(images), labels).backward()
    optimizer.step()
    loss_after = criterion(model(images), labels).item()

    assert loss_after < loss_before
