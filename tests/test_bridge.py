import numpy as np
import pytest

from harmobridge.bridge import hc_to_share, share_to_hc
from harmobridge.field import PrimeField
from harmobridge.harmonic import HarmonicCode, HarmonicConfig
from harmobridge.sharing import AdditiveSharing, ShamirSharing


@pytest.mark.parametrize("kind", ["additive", "shamir"])
def test_bidirectional_bridge(kind):
    field = PrimeField()
    sharing = AdditiveSharing(field) if kind == "additive" else ShamirSharing(field, 7, 2)
    code = HarmonicCode(field, HarmonicConfig(1, 2, 7, (2,)))
    rng = np.random.default_rng(4)
    block = field.array([3, 5, 7])
    mask = field.array([11, 13, 17])
    shared_block = sharing.share(block, rng)
    shared_mask = sharing.share(mask, rng)
    codewords = share_to_hc(code, [shared_block], shared_mask, sharing)
    direct = code.encode([block], mask=mask)
    assert all(field.equal(codewords[label], direct[label]) for label in code.labels)
    outputs = code.evaluate(codewords, lambda value: field.mul(value, value))
    shared_result = hc_to_share(code, outputs, sharing, rng)
    assert field.equal(sharing.reconstruct(shared_result), field.mul(block, block))
