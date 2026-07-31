from __future__ import annotations

from pathlib import Path
from typing import Any


_DATASETS = {"mnist", "cifar10", "cifar100"}


def load_vision_datasets(name: str, root: str | Path, download: bool = True, augmentation: bool = False):
    name = name.lower()
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("Install the vision dependencies before loading datasets") from exc

    root = str(root)
    if name == "mnist":
        train_steps: list[Any] = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return (
            datasets.MNIST(root, train=True, transform=transforms.Compose(train_steps), download=download),
            datasets.MNIST(root, train=False, transform=test_transform, download=download),
        )

    mean = (0.4914, 0.4822, 0.4465) if name == "cifar10" else (0.5071, 0.4867, 0.4408)
    std = (0.2470, 0.2435, 0.2616) if name == "cifar10" else (0.2675, 0.2565, 0.2761)
    train_steps = []
    if augmentation:
        train_steps.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    dataset_class = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    return (
        dataset_class(root, train=True, transform=transforms.Compose(train_steps), download=download),
        dataset_class(root, train=False, transform=test_transform, download=download),
    )


def dataset_labels(dataset) -> list[int]:
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = [int(dataset[index][1]) for index in range(len(dataset))]
    return [int(value) for value in labels]


def dataset_metadata(name: str) -> tuple[int, int]:
    values = {
        "mnist": (1, 10),
        "cifar10": (3, 10),
        "cifar100": (3, 100),
    }
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc
