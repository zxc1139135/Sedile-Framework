from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PrimeField:
    modulus: int = 2**61 - 1

    def array(self, value: Any) -> np.ndarray:
        return np.mod(np.asarray(value, dtype=object), self.modulus)

    def add(self, left: Any, right: Any) -> np.ndarray:
        return self.array(self.array(left) + self.array(right))

    def sub(self, left: Any, right: Any) -> np.ndarray:
        return self.array(self.array(left) - self.array(right))

    def mul(self, left: Any, right: Any) -> np.ndarray:
        return self.array(self.array(left) * self.array(right))

    def neg(self, value: Any) -> np.ndarray:
        return self.array(-self.array(value))

    def inv_scalar(self, value: int) -> int:
        normalized = int(value) % self.modulus
        if normalized == 0:
            raise ZeroDivisionError("Zero has no multiplicative inverse.")
        return pow(normalized, self.modulus - 2, self.modulus)

    def div(self, numerator: Any, denominator: int) -> np.ndarray:
        return self.mul(numerator, self.inv_scalar(denominator))

    def centered(self, value: Any) -> np.ndarray:
        normalized = self.array(value)
        midpoint = (self.modulus - 1) // 2
        return np.where(normalized <= midpoint, normalized, normalized - self.modulus).astype(object)

    def equal(self, left: Any, right: Any) -> bool:
        return bool(np.array_equal(self.array(left), self.array(right)))
