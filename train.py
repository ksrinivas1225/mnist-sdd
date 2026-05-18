import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import load_config
from model import MNISTNet


def get_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    dcfg = cfg["data"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(dcfg["normalize"]["mean"],),
            std=(dcfg["normalize"]["std"],),
        ),
    ])

    train_dataset = datasets.MNIST(
        root=dcfg["root"], train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=dcfg["root"], train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dcfg["batch_size"],
        shuffle=True,
        num_workers=dcfg["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=dcfg["batch_size"],
        shuffle=False,
        num_workers=dcfg["num_workers"],
    )

    return train_loader, test_loader


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(cfg: dict) -> None:
    tcfg = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg["seed"])
    train_loader, test_loader = get_dataloaders(cfg)

    model = MNISTNet(cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    if tcfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=tcfg["lr"])

    checkpoint_path = Path(tcfg["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                avg_loss = running_loss / 100
                print(f"epoch {epoch} | batch {batch_idx:4d} | loss {avg_loss:.4f}")
                running_loss = 0.0

        acc = _accuracy(model, test_loader, device)
        print(f"epoch {epoch} complete | test accuracy: {acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> checkpoint saved (best acc: {best_acc * 100:.2f}%)")


if __name__ == "__main__":
    train(load_config())
