from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from .datasets import dataset_labels, dataset_metadata, load_vision_datasets
from .federated import FLConfig, dirichlet_partition, fedsgd_step
from .field import PrimeField
from .fixedpoint import FixedPoint
from .models import SmallCNN, VGG11NoBN, fixup_resnet20, resnet20_batchnorm
from .sharing import AdditiveSharing


_PAPER_WORKLOADS = {
    ("mnist", "smallcnn"),
    ("cifar10", "vgg11_no_batchnorm"),
    ("cifar100", "vgg11_no_batchnorm"),
    ("cifar10", "fixup_resnet20"),
    ("cifar100", "resnet20_batchnorm"),
}


def validate_paper_workload(dataset: str, model: str) -> None:
    pair = (dataset.lower(), model.lower())
    if pair not in _PAPER_WORKLOADS:
        supported = ", ".join(f"{item[1]}/{item[0]}" for item in sorted(_PAPER_WORKLOADS))
        raise ValueError(f"Unsupported paper workload: {model}/{dataset}. Supported: {supported}")


@dataclass(frozen=True)
class TrainingConfig:
    dataset: str
    model: str
    learning_rate: float | None
    rounds: int = 500
    owners: int = 20
    admitted_per_round: int = 16
    batch_size: int = 128
    dirichlet_alpha: float = 0.3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 2026
    evaluation_interval: int = 10
    augmentation: bool = False
    secure_roundtrip: bool = True
    fractional_bits: int = 16

    def validate(self) -> None:
        if self.learning_rate is None:
            raise ValueError("The manuscript does not state the learning rate. Supply it in the run manifest")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.owners != 20 or self.admitted_per_round != 16 or self.batch_size != 128:
            raise ValueError("Paper-mode runs require 20 owners, 16 admitted owners, and batch size 128")
        if self.rounds != 500:
            raise ValueError("Paper-mode runs require 500 rounds")
        validate_paper_workload(self.dataset, self.model)


def build_model(name: str, dataset: str) -> nn.Module:
    validate_paper_workload(dataset, name)
    channels, classes = dataset_metadata(dataset)
    name = name.lower()
    if name == "smallcnn":
        return SmallCNN(channels=channels, classes=classes)
    if channels != 3:
        raise ValueError(f"{name} expects a three-channel CIFAR input")
    if name == "vgg11_no_batchnorm":
        return VGG11NoBN(classes=classes)
    if name == "fixup_resnet20":
        return fixup_resnet20(classes=classes)
    if name == "resnet20_batchnorm":
        return resnet20_batchnorm(classes=classes)
    raise ValueError(f"Unknown model: {name}")


class ClientBatchStream:
    def __init__(self, dataset: Dataset, indices: np.ndarray, batch_size: int, seed: int) -> None:
        if len(indices) < 1:
            raise ValueError("Client partitions must be non-empty")
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64).copy()
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.order = self.indices.copy()
        self.rng.shuffle(self.order)
        self.cursor = 0

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        selected: list[int] = []
        while len(selected) < self.batch_size:
            remaining = len(self.order) - self.cursor
            take = min(self.batch_size - len(selected), remaining)
            selected.extend(int(value) for value in self.order[self.cursor : self.cursor + take])
            self.cursor += take
            if self.cursor == len(self.order):
                self.order = self.indices.copy()
                self.rng.shuffle(self.order)
                self.cursor = 0
        return default_collate([self.dataset[index] for index in selected])

    def state_dict(self) -> dict[str, object]:
        return {
            "order": self.order.copy(),
            "cursor": self.cursor,
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        order = np.asarray(state["order"], dtype=np.int64)
        if sorted(order.tolist()) != sorted(self.indices.tolist()):
            raise ValueError("Client stream state does not match its partition")
        cursor = int(state["cursor"])
        if not 0 <= cursor < len(order):
            raise ValueError("Invalid client stream cursor")
        self.order = order.copy()
        self.cursor = cursor
        self.rng.bit_generator.state = state["rng"]


def stable_dirichlet_partition(labels: np.ndarray, clients: int, alpha: float, seed: int, minimum_size: int = 1) -> list[np.ndarray]:
    for attempt in range(100):
        parts = dirichlet_partition(labels, clients, alpha, seed + attempt)
        if min(len(part) for part in parts) >= minimum_size:
            return parts
    raise RuntimeError("Could not create non-empty Dirichlet client partitions after 100 attempts")


def secure_gradient_roundtrip(fractional_bits: int = 16, seed: int = 0) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    field = PrimeField()
    fixed = FixedPoint(field, fractional_bits)
    sharing = AdditiveSharing(field, party_count=2)
    rng = np.random.default_rng(seed)

    def transform(gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for name, value in gradients.items():
            array = value.detach().cpu().double().numpy()
            encoded = fixed.encode(array)
            shares = sharing.share(encoded, rng)
            decoded = fixed.decode(sharing.reconstruct(shares))
            output[name] = torch.as_tensor(decoded, device=value.device, dtype=value.dtype)
        return output

    return transform


@torch.inference_mode()
def evaluate(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 256) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        total_loss += float(criterion(logits, targets).cpu())
        correct += int((logits.argmax(dim=1) == targets).sum().cpu())
        total += targets.numel()
    model.train()
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    momentum_buffers: dict[str, torch.Tensor],
    round_index: int,
    rng: np.random.Generator,
    config: TrainingConfig,
    history: list[dict[str, float]],
    streams: list[ClientBatchStream],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "momentum": {name: value.cpu() for name, value in momentum_buffers.items()},
            "round": round_index,
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(),
            "config": asdict(config),
            "history": history,
            "streams": [stream.state_dict() for stream in streams],
        },
        output,
    )


def run_federated_training(
    config: TrainingConfig,
    data_root: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    resume: str | Path | None = None,
) -> dict[str, object]:
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    train_data, test_data = load_vision_datasets(config.dataset, data_root, download=True, augmentation=config.augmentation)
    labels = np.asarray(dataset_labels(train_data), dtype=np.int64)
    partitions = stable_dirichlet_partition(labels, config.owners, config.dirichlet_alpha, config.seed)
    streams = [ClientBatchStream(train_data, indices, config.batch_size, config.seed + index) for index, indices in enumerate(partitions)]
    model = build_model(config.model, config.dataset).to(target_device)
    momentum_buffers: dict[str, torch.Tensor] = {}
    rng = np.random.default_rng(config.seed)
    start_round = 0
    history: list[dict[str, float]] = []
    if resume is not None:
        checkpoint = torch.load(resume, map_location=target_device, weights_only=False)
        if checkpoint.get("config") != asdict(config):
            raise ValueError("Resume checkpoint configuration does not match the requested run")
        model.load_state_dict(checkpoint["model"])
        momentum_buffers = {name: value.to(target_device) for name, value in checkpoint["momentum"].items()}
        start_round = int(checkpoint["round"]) + 1
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        for stream, state in zip(streams, checkpoint["streams"], strict=True):
            stream.load_state_dict(state)
        random.setstate(checkpoint["python_rng"])
        history = list(checkpoint.get("history", []))

    transform = secure_gradient_roundtrip(config.fractional_bits, config.seed) if config.secure_roundtrip else None
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for round_index in range(start_round, config.rounds):
        admitted = rng.choice(config.owners, size=config.admitted_per_round, replace=False)
        batches = [streams[int(client)].next() for client in admitted]
        sample_counts = np.asarray([targets.numel() for _, targets in batches], dtype=np.float64)
        weights = (sample_counts / sample_counts.sum()).tolist()
        fedsgd_step(
            model,
            batches,
            weights,
            learning_rate=float(config.learning_rate),
            momentum_buffers=momentum_buffers,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            gradient_transform=transform,
        )
        if (round_index + 1) % config.evaluation_interval == 0 or round_index + 1 == config.rounds:
            metrics = evaluate(model, test_data, target_device)
            metrics["round"] = float(round_index + 1)
            history.append(metrics)
            with (root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
        save_checkpoint(root / "checkpoint.pt", model, momentum_buffers, round_index, rng, config, history, streams)

    final_metrics = evaluate(model, test_data, target_device)
    summary = {"config": asdict(config), "final": final_metrics, "history": history}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
