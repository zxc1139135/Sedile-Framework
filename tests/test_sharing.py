import numpy as np
import pytest

from harmobridge.field import PrimeField
from harmobridge.sharing import AdditiveSharing, ShamirSharing


@pytest.mark.parametrize("scheme", ["additive", "shamir"])
def test_sharing_round_trip(scheme):
    field = PrimeField()
    sharing = AdditiveSharing(field) if scheme == "additive" else ShamirSharing(field, party_count=7, threshold=2)
    rng = np.random.default_rng(2)
    secret = field.array([1, 2, 3])
    shares = sharing.share(secret, rng)
    recovered = sharing.reconstruct(shares)
    assert field.equal(secret, recovered)
