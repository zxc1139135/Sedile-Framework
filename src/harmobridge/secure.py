from __future__ import annotations

import numpy as np

from .fixedpoint import FixedPoint
from .sharing import LinearSharing, Shares


class SecureSimulator:
    def __init__(self, sharing: LinearSharing, fixed_point: FixedPoint, seed: int = 0) -> None:
        self.sharing = sharing
        self.fixed_point = fixed_point
        self.field = sharing.field
        self.rng = np.random.default_rng(seed)

    def share(self, value) -> Shares:
        return self.sharing.share(self.field.array(value), self.rng)

    def open(self, value: Shares):
        return self.sharing.reconstruct(value)

    def multiply(self, left: Shares, right: Shares) -> Shares:
        shape = left.values[0].shape
        a = self.field.array(self.rng.integers(0, self.field.modulus, size=shape, dtype=np.int64))
        b = self.field.array(self.rng.integers(0, self.field.modulus, size=shape, dtype=np.int64))
        c = self.field.mul(a, b)
        a_share = self.share(a)
        b_share = self.share(b)
        c_share = self.share(c)
        e = self.open(self.sharing.sub(left, a_share))
        d = self.open(self.sharing.sub(right, b_share))
        result = c_share
        result = self.sharing.add(result, Shares(tuple(self.field.mul(e, item) for item in b_share.values)))
        result = self.sharing.add(result, Shares(tuple(self.field.mul(d, item) for item in a_share.values)))
        return self.sharing.add_public(result, self.field.mul(e, d))

    def fixed_multiply(self, left: Shares, right: Shares) -> Shares:
        product = self.multiply(left, right)
        truncated = self.fixed_point.truncate(self.open(product))
        return self.share(truncated)

    def compare_ge(self, left: Shares, right: Shares) -> Shares:
        difference = self.field.centered(self.field.sub(self.open(left), self.open(right)))
        bit = (difference >= 0).astype(object)
        return self.share(bit)

    def select(self, bit: Shares, when_true: Shares, when_false: Shares) -> Shares:
        delta = self.sharing.sub(when_true, when_false)
        return self.sharing.add(when_false, self.multiply(bit, delta))

    def relu(self, value: Shares) -> tuple[Shares, Shares]:
        zero = self.share(np.zeros_like(value.values[0], dtype=object))
        bit = self.compare_ge(value, zero)
        return self.multiply(bit, value), bit

    def clip(self, value: Shares, low: float, high: float) -> Shares:
        opened = self.fixed_point.decode(self.open(value))
        return self.share(self.fixed_point.encode(np.clip(opened, low, high)))

    def max_pool4(self, values: list[Shares]) -> tuple[Shares, list[Shares]]:
        if len(values) != 4:
            raise ValueError("max_pool4 expects four inputs.")
        opened = np.stack([self.fixed_point.decode(self.open(item)) for item in values], axis=0)
        indices = np.argmax(opened, axis=0)
        maximum = np.take_along_axis(opened, indices[None, ...], axis=0)[0]
        route = [self.share((indices == index).astype(object)) for index in range(4)]
        return self.share(self.fixed_point.encode(maximum)), route
