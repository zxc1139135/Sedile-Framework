from __future__ import annotations

import math


def unreplicated_success(omission_probability: float, labels: int = 3) -> float:
    _check_probability(omission_probability)
    return (1.0 - omission_probability) ** labels


def replicated_hc_success(omission_probability: float, replicas_per_label: int = 10, labels: int = 3) -> float:
    _check_probability(omission_probability)
    return (1.0 - omission_probability**replicas_per_label) ** labels


def dual_server_success(omission_probability: float, server_availability: float = 0.999) -> float:
    return replicated_hc_success(omission_probability) * server_availability**2


def committee_success(omission_probability: float, members: int = 7, required: int = 5, member_availability: float = 0.999) -> float:
    _check_probability(omission_probability)
    committee = sum(
        math.comb(members, active) * member_availability**active * (1.0 - member_availability) ** (members - active)
        for active in range(required, members + 1)
    )
    return replicated_hc_success(omission_probability) * committee


def _check_probability(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("Probability must be in [0, 1].")
