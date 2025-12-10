"""
Paillier Cryptosystem Implementation.

Implements the Paillier public-key cryptosystem supporting:
- Homomorphic addition of ciphertexts
- Scalar multiplication of ciphertexts

Used in the similarity-driven client partitioning protocol for
privacy-preserving similarity computation.
"""

import random
import math
from typing import Tuple, Optional
from functools import lru_cache


def _is_prime(n: int, k: int = 25) -> bool:
    """
    Miller-Rabin primality test.
    
    Args:
        n: Number to test for primality
        k: Number of rounds (higher = more accurate)
        
    Returns:
        True if probably prime, False if composite
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Witness loop
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
            
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _generate_prime(bits: int) -> int:
    """Generate a prime number with specified bit length."""
    while True:
        candidate = random.getrandbits(bits)
        candidate |= (1 << bits - 1) | 1  # Ensure correct bit length and odd
        if _is_prime(candidate):
            return candidate


def _lcm(a: int, b: int) -> int:
    """Compute least common multiple of a and b."""
    return abs(a * b) // math.gcd(a, b)


def _L(x: int, n: int) -> int:
    """L function: L(x) = (x - 1) / n."""
    return (x - 1) // n


def _mod_inverse(a: int, m: int) -> int:
    """Compute modular multiplicative inverse using extended Euclidean algorithm."""
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a % m, m)
    return (x % m + m) % m


class PaillierPublicKey:
    """Paillier public key for encryption."""
    
    def __init__(self, n: int, g: int):
        """
        Initialize public key.
        
        Args:
            n: Modulus n = p * q
            g: Generator g in Z*_{n^2}
        """
        self.n = n
        self.g = g
        self.n_sq = n * n
    
    def encrypt(self, plaintext: int, r: Optional[int] = None) -> int:
        """
        Encrypt a plaintext message.
        
        Args:
            plaintext: Message to encrypt (must be in Z_n)
            r: Random value for encryption (generated if not provided)
            
        Returns:
            Ciphertext c = g^m * r^n mod n^2
        """
        if plaintext < 0:
            plaintext = plaintext % self.n
        
        if r is None:
            r = random.randrange(1, self.n)
            while math.gcd(r, self.n) != 1:
                r = random.randrange(1, self.n)
        
        # c = g^m * r^n mod n^2
        g_m = pow(self.g, plaintext, self.n_sq)
        r_n = pow(r, self.n, self.n_sq)
        ciphertext = (g_m * r_n) % self.n_sq
        
        return ciphertext
    
    def add(self, c1: int, c2: int) -> int:
        """
        Homomorphic addition: Dec(c1 * c2) = m1 + m2.
        
        Args:
            c1: First ciphertext
            c2: Second ciphertext
            
        Returns:
            Encrypted sum
        """
        return (c1 * c2) % self.n_sq
    
    def scalar_mult(self, c: int, scalar: int) -> int:
        """
        Scalar multiplication: Dec(c^scalar) = scalar * m.
        
        Args:
            c: Ciphertext
            scalar: Scalar multiplier
            
        Returns:
            Encrypted product
        """
        if scalar < 0:
            scalar = scalar % self.n
        return pow(c, scalar, self.n_sq)


class PaillierPrivateKey:
    """Paillier private key for decryption."""
    
    def __init__(self, lam: int, mu: int, n: int):
        """
        Initialize private key.
        
        Args:
            lam: Lambda = lcm(p-1, q-1)
            mu: Mu = L(g^lambda mod n^2)^{-1} mod n
            n: Modulus
        """
        self.lam = lam
        self.mu = mu
        self.n = n
        self.n_sq = n * n
    
    def decrypt(self, ciphertext: int) -> int:
        """
        Decrypt a ciphertext.
        
        Args:
            ciphertext: Encrypted message
            
        Returns:
            Decrypted plaintext in Z_n
        """
        # m = L(c^lambda mod n^2) * mu mod n
        c_lam = pow(ciphertext, self.lam, self.n_sq)
        L_val = _L(c_lam, self.n)
        plaintext = (L_val * self.mu) % self.n
        
        # Handle negative numbers (values > n/2 are treated as negative)
        if plaintext > self.n // 2:
            plaintext = plaintext - self.n
            
        return plaintext


class PaillierKeyPair:
    """Container for Paillier key pair."""
    
    def __init__(self, public_key: PaillierPublicKey, private_key: PaillierPrivateKey):
        self.public = public_key
        self.private = private_key


def generate_keypair(key_size: int = 2048) -> PaillierKeyPair:
    """
    Generate a new Paillier key pair.
    
    Args:
        key_size: Bit length of n (modulus)
        
    Returns:
        PaillierKeyPair containing public and private keys
    """
    # Generate two distinct primes p and q
    p_bits = key_size // 2
    p = _generate_prime(p_bits)
    q = _generate_prime(p_bits)
    while q == p:
        q = _generate_prime(p_bits)
    
    n = p * q
    n_sq = n * n
    
    # Compute lambda = lcm(p-1, q-1)
    lam = _lcm(p - 1, q - 1)
    
    # Select g = n + 1 (simplified choice that works well)
    g = n + 1
    
    # Compute mu = L(g^lambda mod n^2)^{-1} mod n
    g_lam = pow(g, lam, n_sq)
    L_val = _L(g_lam, n)
    mu = _mod_inverse(L_val, n)
    
    public_key = PaillierPublicKey(n, g)
    private_key = PaillierPrivateKey(lam, mu, n)
    
    return PaillierKeyPair(public_key, private_key)


# Lightweight key generation for testing (smaller key size)
def generate_keypair_fast(key_size: int = 512) -> PaillierKeyPair:
    """Generate key pair with smaller key size for faster testing."""
    return generate_keypair(key_size)


if __name__ == '__main__':
    # Basic functionality test
    print("Testing Paillier encryption...")
    
    keys = generate_keypair_fast(512)
    pk, sk = keys.public, keys.private
    
    # Test encryption/decryption
    m1, m2 = 42, 58
    c1 = pk.encrypt(m1)
    c2 = pk.encrypt(m2)
    
    assert sk.decrypt(c1) == m1, "Decryption failed"
    assert sk.decrypt(c2) == m2, "Decryption failed"
    
    # Test homomorphic addition
    c_sum = pk.add(c1, c2)
    assert sk.decrypt(c_sum) == m1 + m2, "Homomorphic addition failed"
    
    # Test scalar multiplication
    scalar = 3
    c_mult = pk.scalar_mult(c1, scalar)
    assert sk.decrypt(c_mult) == m1 * scalar, "Scalar multiplication failed"
    
    # Test negative numbers
    m_neg = -17
    c_neg = pk.encrypt(m_neg)
    assert sk.decrypt(c_neg) == m_neg, "Negative number handling failed"
    
    print("All tests passed!")
