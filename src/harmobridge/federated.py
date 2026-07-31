from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FLConfig:
    clients: int = 20
    admitted_per_round: int = 16
    batch_size: int = 128
    dirichlet_alpha: float = 0.3
    rounds: int = 500
    momentum: float = 0.9
    weight_decay: float = 5e-4


def dirichlet_partition(labels: np.ndarray, client_count: int = 20, alpha: float = 0.3, seed: int = 0) -> list[np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[] for _ in range(client_count)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(client_count, alpha))
        cut_points = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        for client, split in enumerate(np.split(indices, cut_points)):
            partitions[client].extend(split.tolist())
    return [np.asarray(sorted(values), dtype=np.int64) for values in partitions]


def fedsgd_step(model: nn.Module, client_batches: list[tuple[torch.Tensor, torch.Tensor]], sample_weights: list[float], learning_rate: float, momentum_buffers: dict[str, torch.Tensor], momentum: float = 0.9, weight_decay: float = 5e-4, gradient_transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None) -> None:
    if len(client_batches) != len(sample_weights):
        raise ValueError("Each client batch needs one public sample weight.")
    if not np.isclose(sum(sample_weights), 1.0, atol=1e-6):
        raise ValueError("Public sample weights must sum to one.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    device = next(model.parameters()).device
    aggregate = {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}
    criterion = nn.CrossEntropyLoss()
    for (inputs, targets), weight in zip(client_batches, sample_weights, strict=True):
        model.zero_grad(set_to_none=True)
        logits = model(inputs.to(device))
        loss = criterion(logits, targets.to(device))
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                aggregate[name].add_(parameter.grad, alpha=float(weight))
    if gradient_transform is not None:
        aggregate = gradient_transform(aggregate)
        if set(aggregate) != {name for name, _ in model.named_parameters()}:
            raise ValueError("gradient_transform changed the parameter keys")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            gradient = aggregate[name] + weight_decay * parameter
            buffer = momentum_buffers.setdefault(name, torch.zeros_like(parameter))
            buffer.mul_(momentum).add_(gradient)
            parameter.add_(buffer, alpha=-learning_rate)
