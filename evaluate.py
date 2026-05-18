import torch

from config import load_config
from model import MNISTNet
from train import get_dataloaders


def evaluate(cfg: dict) -> float:
    ecfg = cfg["evaluate"]
    device = torch.device(ecfg["device"])

    _, test_loader = get_dataloaders(cfg)

    model = MNISTNet(cfg).to(device)
    model.load_state_dict(torch.load(ecfg["checkpoint_path"], map_location=device))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"test accuracy: {acc * 100:.2f}%  ({correct}/{total})")
    return acc


if __name__ == "__main__":
    evaluate(load_config())
