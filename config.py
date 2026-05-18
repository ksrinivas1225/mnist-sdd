import yaml
from pathlib import Path

_REQUIRED_KEYS = {
    "data": ["root", "batch_size", "normalize"],
    "model": ["hidden_layers", "activation"],
    "train": ["epochs", "optimizer", "lr", "checkpoint_path"],
    "evaluate": ["checkpoint_path", "device"],
}


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for section, keys in _REQUIRED_KEYS.items():
        if section not in cfg:
            raise ValueError(f"config.yaml is missing required section '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"config.yaml [{section}] is missing required key '{key}'")

    return cfg


if __name__ == "__main__":
    import pprint
    pprint.pprint(load_config())
