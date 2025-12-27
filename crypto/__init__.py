"""
Cryptographic primitives for Sedile framework.
"""

from .paillier import (
    PaillierPublicKey,
    PaillierPrivateKey,
    PaillierKeyPair,
    generate_keypair,
    generate_keypair_fast
)

from .shamir import (
    ShamirSecretSharing,
    SecretShareDistributor,
    quantize_to_field,
    dequantize_from_field
)

from .harmonic import (
    HarmonicCoder,
    HarmonicEncoder,
    GradientDecoder,
    create_harmonic_system
)

__all__ = [
    # Paillier
    'PaillierPublicKey',
    'PaillierPrivateKey',
    'PaillierKeyPair',
    'generate_keypair',
    'generate_keypair_fast',
    # Shamir
    'ShamirSecretSharing',
    'SecretShareDistributor',
    'quantize_to_field',
    'dequantize_from_field',
    # Harmonic
    'HarmonicCoder',
    'HarmonicEncoder',
    'GradientDecoder',
    'create_harmonic_system'
]
