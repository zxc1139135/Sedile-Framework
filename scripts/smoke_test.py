import numpy as np

from harmobridge.bridge import hc_to_share, share_to_hc
from harmobridge.field import PrimeField
from harmobridge.harmonic import HarmonicCode, HarmonicConfig
from harmobridge.sharing import AdditiveSharing


def main() -> None:
    field = PrimeField()
    sharing = AdditiveSharing(field)
    code = HarmonicCode(field, HarmonicConfig(blocks=1, degree=2, c=7, betas=(2,)))
    rng = np.random.default_rng(8)
    logical = field.array([2, 3, 5])
    mask = field.array([7, 11, 13])
    codewords = share_to_hc(code, [sharing.share(logical, rng)], sharing.share(mask, rng), sharing)
    outputs = code.evaluate(codewords, lambda value: field.add(field.mul(value, value), 3))
    shared = hc_to_share(code, outputs, sharing, rng)
    decoded = sharing.reconstruct(shared)
    expected = field.add(field.mul(logical, logical), 3)
    print({"correct": field.equal(decoded, expected), "decoded": [int(x) for x in decoded]})


if __name__ == "__main__":
    main()
