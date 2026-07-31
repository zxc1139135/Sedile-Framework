from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .field import PrimeField


@dataclass(frozen=True)
class HarmonicConfig:
    blocks: int
    degree: int
    c: int
    betas: tuple[int, ...]


class HarmonicCode:
    def __init__(self, field: PrimeField, config: HarmonicConfig) -> None:
        self.field = field
        self.config = config
        self._validate()

    @property
    def labels(self) -> tuple[str, ...]:
        groups = tuple(f"G{i}_{j}" for i in range(1, self.config.degree) for j in range(1, self.config.blocks + 1))
        return ("L", "R") + groups

    def encode(self, blocks: list[np.ndarray], mask: np.ndarray | None = None, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        if len(blocks) != self.config.blocks:
            raise ValueError("The logical block count does not match the code configuration.")
        values = [self.field.array(block) for block in blocks]
        if any(value.shape != values[0].shape for value in values):
            raise ValueError("All logical blocks must have the same shape.")
        if mask is None:
            if rng is None:
                raise ValueError("rng is required when mask is not provided.")
            mask = self.field.array(rng.integers(0, self.field.modulus, size=values[0].shape, dtype=np.int64))
        else:
            mask = self.field.array(mask)

        c = self.config.c
        prefixes = [mask]
        running = self.field.array(0)
        for j, block in enumerate(values, start=1):
            running = self.field.add(running, block)
            numerator = self.field.sub(self.field.mul(c, mask), running)
            prefixes.append(self.field.div(numerator, c - j))

        output: dict[str, np.ndarray] = {"L": prefixes[0], "R": prefixes[-1]}
        for i, beta in enumerate(self.config.betas, start=1):
            for j, block in enumerate(values, start=1):
                theta = (beta * (c - j + 1) * self.field.inv_scalar(c)) % self.field.modulus
                one_minus = (1 - theta) % self.field.modulus
                output[f"G{i}_{j}"] = self.field.add(
                    self.field.mul(one_minus, block),
                    self.field.mul(theta, prefixes[j - 1]),
                )
        return output

    def decoder_coefficients(self) -> dict[str, int]:
        c = self.config.c
        coefficients = {
            "L": self._h(c),
            "R": (-self._h(c - self.config.blocks)) % self.field.modulus,
        }
        for j in range(1, self.config.blocks + 1):
            a = ((c - j + 1) * self.field.inv_scalar(c - j)) % self.field.modulus
            thetas = [
                (beta * (c - j + 1) * self.field.inv_scalar(c)) % self.field.modulus
                for beta in self.config.betas
            ]
            nodes = [1, a] + thetas
            for i, theta in enumerate(thetas, start=1):
                coefficients[f"G{i}_{j}"] = self._lagrange_at_zero(theta, nodes)
        return coefficients

    def evaluate(self, codewords: dict[str, np.ndarray], function: Callable[[np.ndarray], np.ndarray]) -> dict[str, np.ndarray]:
        missing = set(self.labels) - set(codewords)
        if missing:
            raise ValueError(f"Missing codewords: {sorted(missing)}")
        return {label: self.field.array(function(codewords[label])) for label in self.labels}

    def decode(self, worker_outputs: dict[str, np.ndarray]) -> np.ndarray:
        coefficients = self.decoder_coefficients()
        total = self.field.array(0)
        for label in self.labels:
            if label not in worker_outputs:
                raise ValueError(f"Responder set is not qualified; missing {label}.")
            total = self.field.add(total, self.field.mul(coefficients[label], worker_outputs[label]))
        return total

    def _h(self, value: int) -> int:
        result = value % self.field.modulus
        for beta in self.config.betas:
            numerator = (beta * value) % self.field.modulus
            denominator = (beta * value - self.config.c) % self.field.modulus
            result = (result * numerator) % self.field.modulus
            result = (result * self.field.inv_scalar(denominator)) % self.field.modulus
        return result

    def _lagrange_at_zero(self, node: int, nodes: list[int]) -> int:
        result = 1
        for other in nodes:
            if other == node:
                continue
            result = (result * (-other)) % self.field.modulus
            result = (result * self.field.inv_scalar(node - other)) % self.field.modulus
        return result

    def _validate(self) -> None:
        config = self.config
        if config.blocks < 1 or config.degree < 1:
            raise ValueError("blocks and degree must be positive.")
        if len(config.betas) != config.degree - 1:
            raise ValueError("The code requires degree-1 beta values.")
        forbidden_c = {value % self.field.modulus for value in range(config.blocks + 1)}
        if config.c % self.field.modulus in forbidden_c:
            raise ValueError("c must differ from 0 through K.")
        if len(set(beta % self.field.modulus for beta in config.betas)) != len(config.betas):
            raise ValueError("beta values must be distinct.")
        for beta in config.betas:
            if beta % self.field.modulus == 0:
                raise ValueError("beta values must be nonzero.")
            for j in range(config.blocks + 1):
                forbidden = (config.c * self.field.inv_scalar(config.c - j)) % self.field.modulus
                if beta % self.field.modulus == forbidden:
                    raise ValueError("A beta value creates a zero interpolation denominator.")
