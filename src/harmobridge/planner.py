from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Operation:
    name: str
    input_degrees: tuple[int, ...]


@dataclass(frozen=True)
class Segment:
    domain: str
    operations: tuple[Operation, ...]
    output_degree: int


_HARMONIC_LINEAR = {"affine", "conv", "linear", "add", "residual_add", "public_average"}
_SHARE_DOMAIN = {"relu", "leaky_relu", "clip", "max_pool", "trunc", "batch_norm", "softmax_ce", "comparison", "reciprocal", "sqrt"}


def output_degree(operation: Operation) -> int:
    if operation.name in _HARMONIC_LINEAR:
        return max(operation.input_degrees, default=0)
    if operation.name in {"multiply", "outer_product"}:
        return sum(operation.input_degrees)
    if operation.name in _SHARE_DOMAIN:
        return 1
    raise ValueError(f"Unsupported operation: {operation.name}")


def compile_segments(operations: list[Operation], degree_budget: int) -> list[Segment]:
    segments: list[Segment] = []
    current_domain = None
    current: list[Operation] = []
    current_degree = 1
    for operation in operations:
        degree = output_degree(operation)
        domain = "share" if operation.name in _SHARE_DOMAIN or degree > degree_budget else "harmonic"
        if current_domain is not None and domain != current_domain:
            segments.append(Segment(current_domain, tuple(current), current_degree))
            current = []
        current_domain = domain
        current.append(operation)
        current_degree = degree
    if current:
        segments.append(Segment(current_domain or "harmonic", tuple(current), current_degree))
    return segments
