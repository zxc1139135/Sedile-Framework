from __future__ import annotations

import numpy as np

from .harmonic import HarmonicCode
from .sharing import LinearSharing, Shares, sum_shares


def hc_to_share(
    code: HarmonicCode,
    worker_outputs: dict[str, np.ndarray],
    sharing: LinearSharing,
    rng: np.random.Generator,
) -> Shares:
    coefficients = code.decoder_coefficients()
    contributions = []
    for label in code.labels:
        if label not in worker_outputs:
            raise ValueError(f"Responder set is not qualified; missing {label}.")
        value = code.field.mul(coefficients[label], worker_outputs[label])
        contributions.append(sharing.share(value, rng))
    return sum_shares(sharing, contributions)


def share_to_hc(
    code: HarmonicCode,
    shared_blocks: list[Shares],
    shared_mask: Shares,
    sharing: LinearSharing,
) -> dict[str, np.ndarray]:
    if len(shared_blocks) != code.config.blocks:
        raise ValueError("The logical block count does not match the code configuration.")
    c = code.config.c
    prefix_shares = [shared_mask]
    running = sharing.scale(shared_blocks[0], 0)
    for j, block in enumerate(shared_blocks, start=1):
        running = sharing.add(running, block)
        numerator = sharing.sub(sharing.scale(shared_mask, c), running)
        prefix_shares.append(sharing.scale(numerator, code.field.inv_scalar(c - j)))

    output = {
        "L": sharing.reconstruct(prefix_shares[0]),
        "R": sharing.reconstruct(prefix_shares[-1]),
    }
    for i, beta in enumerate(code.config.betas, start=1):
        for j, block in enumerate(shared_blocks, start=1):
            theta = (beta * (c - j + 1) * code.field.inv_scalar(c)) % code.field.modulus
            combined = sharing.add(
                sharing.scale(block, (1 - theta) % code.field.modulus),
                sharing.scale(prefix_shares[j - 1], theta),
            )
            output[f"G{i}_{j}"] = sharing.reconstruct(combined)
    return output
