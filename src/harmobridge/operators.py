from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .fixedpoint import FixedPoint


@dataclass(frozen=True)
class PolynomialSegment:
    low: float
    high: float
    coefficients: tuple[float, ...]


def fit_piecewise(function: Callable[[np.ndarray], np.ndarray], boundaries: list[float], degree: int = 3, samples: int = 512) -> list[PolynomialSegment]:
    if len(boundaries) < 2:
        raise ValueError("At least two boundaries are required.")
    segments = []
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        x = np.linspace(low, high, samples)
        coefficients = tuple(float(value) for value in np.polynomial.polynomial.polyfit(x, function(x), degree))
        segments.append(PolynomialSegment(low, high, coefficients))
    return segments


def evaluate_piecewise(value: np.ndarray, segments: list[PolynomialSegment]) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    low = segments[0].low
    high = segments[-1].high
    if np.any(values < low) or np.any(values > high):
        raise ValueError("Input is outside the declared approximation domain.")
    output = np.empty_like(values)
    for index, segment in enumerate(segments):
        mask = (values >= segment.low) & (values < segment.high)
        if index == len(segments) - 1:
            mask = (values >= segment.low) & (values <= segment.high)
        output[mask] = np.polynomial.polynomial.polyval(values[mask], segment.coefficients)
    return output


def reciprocal_newton(value: np.ndarray, initial: np.ndarray, iterations: int = 2) -> np.ndarray:
    estimate = np.asarray(initial, dtype=np.float64)
    target = np.asarray(value, dtype=np.float64)
    for _ in range(iterations):
        estimate = estimate * (2.0 - target * estimate)
    return estimate


def inverse_sqrt_newton(value: np.ndarray, initial: np.ndarray, iterations: int = 2) -> np.ndarray:
    estimate = np.asarray(initial, dtype=np.float64)
    target = np.asarray(value, dtype=np.float64)
    for _ in range(iterations):
        estimate = 0.5 * estimate * (3.0 - target * estimate * estimate)
    return estimate


def softmax_adjoint(logits: np.ndarray, labels: np.ndarray, exp_segments: list[PolynomialSegment], reciprocal_segments: list[PolynomialSegment], iterations: int = 2) -> np.ndarray:
    shifted = np.clip(logits - np.max(logits, axis=-1, keepdims=True), -8.0, 0.0)
    exponentials = evaluate_piecewise(shifted, exp_segments)
    sums = exponentials.sum(axis=-1, keepdims=True)
    initial = evaluate_piecewise(sums, reciprocal_segments)
    inverse = reciprocal_newton(sums, initial, iterations=iterations)
    probabilities = exponentials * inverse
    return probabilities - labels


def batch_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float, initializer: list[PolynomialSegment], iterations: int = 2) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    mean = x.mean(axis=0, keepdims=True)
    variance = (x * x).mean(axis=0, keepdims=True) - mean * mean
    a = variance + epsilon
    initial = evaluate_piecewise(a, initializer)
    inverse = inverse_sqrt_newton(a, initial, iterations=iterations)
    normalized = (x - mean) * inverse
    return gamma * normalized + beta, (normalized, inverse)
