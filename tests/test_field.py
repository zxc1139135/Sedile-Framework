import numpy as np

from harmobridge.field import PrimeField
from harmobridge.fixedpoint import FixedPoint


def test_fixed_point_multiplication():
    field = PrimeField()
    fixed = FixedPoint(field, fractional_bits=16)
    left = fixed.encode(np.array([-1.25, 2.0]))
    right = fixed.encode(np.array([2.0, 0.5]))
    decoded = fixed.decode(fixed.multiply(left, right))
    np.testing.assert_allclose(decoded, [-2.5, 1.0], atol=2 / fixed.scale)
