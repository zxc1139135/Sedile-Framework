from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .field import PrimeField


@dataclass(frozen=True)
class FixedPoint:
    field: PrimeField
    fractional_bits: int = 16

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    def encode(self, value: Any) -> np.ndarray:
        scaled = np.rint(np.asarray(value, dtype=np.float64) * self.scale).astype(object)
        return self.field.array(scaled)

    def decode(self, value: Any) -> np.ndarray:
        centered = self.field.centered(value).astype(np.float64)
        return centered / self.scale

    def truncate(self, raw_product: Any) -> np.ndarray:
        centered = self.field.centered(raw_product)
        truncated = np.vectorize(lambda item: int(item) // self.scale, otypes=[object])(centered)
        return self.field.array(truncated)

    def multiply(self, left: Any, right: Any) -> np.ndarray:
        return self.truncate(self.field.mul(left, right))

    def clip(self, value: Any, low: float, high: float) -> np.ndarray:
        decoded = self.decode(value)
        return self.encode(np.clip(decoded, low, high))
