import torch
import torch.nn as nn

from config import load_config

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MNISTNet(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg = cfg["model"]
        act_cls = _ACTIVATIONS[mcfg["activation"]]
        layer_sizes = [784] + mcfg["hidden_layers"] + [10]

        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 10:
                layers.append(act_cls())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


if __name__ == "__main__":
    cfg = load_config()
    model = MNISTNet(cfg)
    print(model)
    out = model(torch.zeros(4, 1, 28, 28))
    print(f"output shape: {tuple(out.shape)}")
    assert out.shape == (4, 10), "shape mismatch"
    print("OK")
