import numpy as np
import pytest

from harmobridge.field import PrimeField
from harmobridge.harmonic import HarmonicCode, HarmonicConfig


@pytest.mark.parametrize("blocks,degree,betas", [(1, 2, (2,)), (2, 3, (2, 3))])
def test_harmonic_decodes_polynomial_sum(blocks, degree, betas):
    field = PrimeField()
    code = HarmonicCode(field, HarmonicConfig(blocks=blocks, degree=degree, c=11, betas=betas))
    rng = np.random.default_rng(3)
    logical = [field.array(rng.integers(0, 100, size=(4,), dtype=np.int64)) for _ in range(blocks)]
    codewords = code.encode(logical, rng=rng)

    def polynomial(x):
        result = field.array(5)
        power = field.array(1)
        for coefficient in range(1, degree + 1):
            power = field.mul(power, x)
            result = field.add(result, field.mul(coefficient + 1, power))
        return result

    worker_outputs = code.evaluate(codewords, polynomial)
    decoded = code.decode(worker_outputs)
    expected = field.array(0)
    for block in logical:
        expected = field.add(expected, polynomial(block))
    assert field.equal(decoded, expected)
