from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .field import PrimeField


@dataclass(frozen=True)
class Shares:
    values: tuple[np.ndarray, ...]

    @property
    def party_count(self) -> int:
        return len(self.values)


class LinearSharing(Protocol):
    field: PrimeField
    party_count: int
    def share(self, secret, rng: np.random.Generator) -> Shares: ...
    def reconstruct(self, shares: Shares, parties: tuple[int, ...] | None = None) -> np.ndarray: ...
    def add(self, left: Shares, right: Shares) -> Shares: ...
    def sub(self, left: Shares, right: Shares) -> Shares: ...
    def scale(self, shares: Shares, scalar: int) -> Shares: ...
    def add_public(self, shares: Shares, value) -> Shares: ...


class AdditiveSharing:
    def __init__(self, field: PrimeField, party_count: int = 2) -> None:
        if party_count < 2:
            raise ValueError("Additive sharing requires at least two parties.")
        self.field = field
        self.party_count = party_count

    def share(self, secret, rng: np.random.Generator) -> Shares:
        target = self.field.array(secret)
        random_parts = [self.field.array(rng.integers(0, self.field.modulus, size=target.shape, dtype=np.int64)) for _ in range(self.party_count - 1)]
        total = self.field.array(0)
        for part in random_parts:
            total = self.field.add(total, part)
        final = self.field.sub(target, total)
        return Shares(tuple(random_parts + [final]))

    def reconstruct(self, shares: Shares, parties: tuple[int, ...] | None = None) -> np.ndarray:
        if parties is not None and len(parties) != self.party_count:
            raise ValueError("All additive shares are required for reconstruction.")
        total = self.field.array(0)
        for value in shares.values:
            total = self.field.add(total, value)
        return total

    def add(self, left: Shares, right: Shares) -> Shares:
        _check_shape(left, right)
        return Shares(tuple(self.field.add(a, b) for a, b in zip(left.values, right.values, strict=True)))

    def sub(self, left: Shares, right: Shares) -> Shares:
        _check_shape(left, right)
        return Shares(tuple(self.field.sub(a, b) for a, b in zip(left.values, right.values, strict=True)))

    def scale(self, shares: Shares, scalar: int) -> Shares:
        return Shares(tuple(self.field.mul(value, scalar) for value in shares.values))

    def add_public(self, shares: Shares, value) -> Shares:
        values = list(shares.values)
        values[0] = self.field.add(values[0], value)
        return Shares(tuple(values))


class ShamirSharing:
    def __init__(self, field: PrimeField, party_count: int = 7, threshold: int = 2) -> None:
        if threshold < 1 or party_count < threshold + 1:
            raise ValueError("Invalid Shamir threshold or party count.")
        self.field = field
        self.party_count = party_count
        self.threshold = threshold
        self.points = tuple(range(1, party_count + 1))

    def share(self, secret, rng: np.random.Generator) -> Shares:
        target = self.field.array(secret)
        coefficients = [target]
        for _ in range(self.threshold):
            coefficients.append(self.field.array(rng.integers(0, self.field.modulus, size=target.shape, dtype=np.int64)))
        values = []
        for point in self.points:
            result = self.field.array(0)
            power = 1
            for coefficient in coefficients:
                result = self.field.add(result, self.field.mul(coefficient, power))
                power = (power * point) % self.field.modulus
            values.append(result)
        return Shares(tuple(values))

    def reconstruct(self, shares: Shares, parties: tuple[int, ...] | None = None) -> np.ndarray:
        selected = parties or tuple(range(self.threshold + 1))
        if len(selected) < self.threshold + 1:
            raise ValueError("Not enough Shamir shares for reconstruction.")
        total = self.field.array(0)
        selected_points = [self.points[index] for index in selected]
        for index, point in zip(selected, selected_points, strict=True):
            coefficient = 1
            for other in selected_points:
                if other == point:
                    continue
                coefficient = (coefficient * (-other)) % self.field.modulus
                coefficient = (coefficient * self.field.inv_scalar(point - other)) % self.field.modulus
            total = self.field.add(total, self.field.mul(shares.values[index], coefficient))
        return total

    def add(self, left: Shares, right: Shares) -> Shares:
        _check_shape(left, right)
        return Shares(tuple(self.field.add(a, b) for a, b in zip(left.values, right.values, strict=True)))

    def sub(self, left: Shares, right: Shares) -> Shares:
        _check_shape(left, right)
        return Shares(tuple(self.field.sub(a, b) for a, b in zip(left.values, right.values, strict=True)))

    def scale(self, shares: Shares, scalar: int) -> Shares:
        return Shares(tuple(self.field.mul(value, scalar) for value in shares.values))

    def add_public(self, shares: Shares, value) -> Shares:
        return Shares(tuple(self.field.add(item, value) for item in shares.values))


def sum_shares(sharing: LinearSharing, values: list[Shares]) -> Shares:
    if not values:
        raise ValueError("At least one sharing is required.")
    result = values[0]
    for item in values[1:]:
        result = sharing.add(result, item)
    return result


def _check_shape(left: Shares, right: Shares) -> None:
    if left.party_count != right.party_count:
        raise ValueError("Share vectors use different party counts.")
