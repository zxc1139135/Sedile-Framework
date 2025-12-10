"""
Shamir Secret Sharing Implementation.

Implements (T, N)-threshold secret sharing scheme where:
- A secret is split into N shares
- Any T or more shares can reconstruct the secret
- Fewer than T shares reveal nothing about the secret

Used in the intra-partition data sharing protocol for
privacy-preserving dataset distribution among clients.
"""

import numpy as np
from typing import List, Tuple, Optional
import random


def _mod_inverse(a: int, p: int) -> int:
    """
    Compute modular multiplicative inverse using Fermat's little theorem.
    
    For prime p: a^{-1} = a^{p-2} mod p
    """
    return pow(a, p - 2, p)


def _lagrange_coefficient(i: int, points: List[int], p: int) -> int:
    """
    Compute Lagrange coefficient for point i.
    
    Args:
        i: Index of the point
        points: List of all evaluation points
        p: Prime modulus
        
    Returns:
        Lagrange coefficient for reconstruction at x=0
    """
    xi = points[i]
    numerator = 1
    denominator = 1
    
    for j, xj in enumerate(points):
        if i != j:
            numerator = (numerator * (-xj)) % p
            denominator = (denominator * (xi - xj)) % p
    
    return (numerator * _mod_inverse(denominator, p)) % p


class ShamirSecretSharing:
    """
    Shamir (T, N)-threshold secret sharing scheme.
    
    Attributes:
        threshold: Minimum number of shares needed for reconstruction
        num_shares: Total number of shares to generate
        prime: Prime modulus for finite field arithmetic
    """
    
    def __init__(self, threshold: int, num_shares: int, prime: int):
        """
        Initialize the secret sharing scheme.
        
        Args:
            threshold: T - minimum shares for reconstruction
            num_shares: N - total number of shares
            prime: Large prime for finite field operations
        """
        if threshold > num_shares:
            raise ValueError("Threshold cannot exceed number of shares")
        if threshold < 2:
            raise ValueError("Threshold must be at least 2")
            
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = prime
        
        # Pre-compute evaluation points (1, 2, ..., N)
        self.eval_points = list(range(1, num_shares + 1))
    
    def _generate_polynomial_coeffs(self, secret: int) -> List[int]:
        """
        Generate random polynomial coefficients.
        
        Polynomial: f(x) = secret + a_1*x + a_2*x^2 + ... + a_{T-1}*x^{T-1}
        """
        coeffs = [secret]
        for _ in range(self.threshold - 1):
            coeffs.append(random.randint(1, self.prime - 1))
        return coeffs
    
    def _evaluate_polynomial(self, coeffs: List[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method."""
        result = 0
        for coeff in reversed(coeffs):
            result = (result * x + coeff) % self.prime
        return result
    
    def share_secret(self, secret: int) -> List[Tuple[int, int]]:
        """
        Split a secret into N shares.
        
        Args:
            secret: The secret value to share (must be in Z_p)
            
        Returns:
            List of (point, share) tuples
        """
        secret = secret % self.prime
        coeffs = self._generate_polynomial_coeffs(secret)
        
        shares = []
        for x in self.eval_points:
            y = self._evaluate_polynomial(coeffs, x)
            shares.append((x, y))
        
        return shares
    
    def share_matrix(self, matrix: np.ndarray) -> List[np.ndarray]:
        """
        Share a matrix element-wise.
        
        Args:
            matrix: 2D numpy array to share
            
        Returns:
            List of N share matrices
        """
        original_shape = matrix.shape
        flat = matrix.flatten()
        
        # Initialize share storage
        share_matrices = [np.zeros(len(flat), dtype=np.int64) for _ in range(self.num_shares)]
        
        # Share each element
        for idx, val in enumerate(flat):
            shares = self.share_secret(int(val) % self.prime)
            for share_idx, (_, share_val) in enumerate(shares):
                share_matrices[share_idx][idx] = share_val
        
        # Reshape to original shape
        return [sm.reshape(original_shape) for sm in share_matrices]
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct secret from T or more shares using Lagrange interpolation.
        
        Args:
            shares: List of (point, share) tuples
            
        Returns:
            Reconstructed secret
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares, got {len(shares)}")
        
        # Use only threshold number of shares
        shares = shares[:self.threshold]
        points = [s[0] for s in shares]
        values = [s[1] for s in shares]
        
        # Lagrange interpolation at x=0
        secret = 0
        for i in range(len(shares)):
            coeff = _lagrange_coefficient(i, points, self.prime)
            secret = (secret + values[i] * coeff) % self.prime
        
        return secret
    
    def reconstruct_matrix(self, share_matrices: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct a matrix from share matrices.
        
        Args:
            share_matrices: List of T or more share matrices
            
        Returns:
            Reconstructed original matrix
        """
        if len(share_matrices) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")
        
        original_shape = share_matrices[0].shape
        num_elements = share_matrices[0].size
        
        result = np.zeros(num_elements, dtype=np.int64)
        
        for idx in range(num_elements):
            shares = [(i + 1, int(share_matrices[i].flat[idx])) 
                     for i in range(len(share_matrices))]
            result[idx] = self.reconstruct_secret(shares)
        
        return result.reshape(original_shape)
    
    def add_shares(self, share1: int, share2: int) -> int:
        """
        Add two shares (homomorphic property).
        
        The result can be used to reconstruct the sum of secrets.
        """
        return (share1 + share2) % self.prime
    
    def add_share_matrices(self, matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        """Add two share matrices element-wise."""
        return (matrix1.astype(np.int64) + matrix2.astype(np.int64)) % self.prime
    
    def scalar_mult_share(self, share: int, scalar: int) -> int:
        """Multiply a share by a scalar (homomorphic property)."""
        return (share * scalar) % self.prime
    
    def scalar_mult_matrix(self, matrix: np.ndarray, scalar: int) -> np.ndarray:
        """Multiply a share matrix by a scalar."""
        return (matrix.astype(np.int64) * scalar) % self.prime


class SecretShareDistributor:
    """
    Handles distribution of secret shares among clients in a partition.
    
    Each client shares its dataset with all other clients in the partition,
    and receives shares from all other clients.
    """
    
    def __init__(self, num_clients: int, threshold: int, prime: int):
        """
        Initialize the distributor.
        
        Args:
            num_clients: Number of clients in partition
            threshold: T-privacy threshold
            prime: Prime modulus
        """
        self.num_clients = num_clients
        self.threshold = threshold
        self.prime = prime
        self.ss = ShamirSecretSharing(threshold, num_clients, prime)
    
    def generate_shares_for_client(
        self, 
        client_id: int, 
        data: np.ndarray
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Generate shares of a client's data for all other clients.
        
        Args:
            client_id: ID of the sharing client
            data: Client's dataset to share
            
        Returns:
            List of (recipient_id, share_matrix) tuples
        """
        share_matrices = self.ss.share_matrix(data)
        
        result = []
        for recipient_id in range(self.num_clients):
            result.append((recipient_id, share_matrices[recipient_id]))
        
        return result
    
    def combine_received_shares(
        self, 
        received_shares: List[Tuple[int, np.ndarray]]
    ) -> np.ndarray:
        """
        Combine shares received from all clients in partition.
        
        Args:
            received_shares: List of (sender_id, share_matrix) tuples
            
        Returns:
            Combined share matrix representing global partition data
        """
        # Sort by sender_id and concatenate along data dimension
        received_shares = sorted(received_shares, key=lambda x: x[0])
        combined = np.concatenate([share for _, share in received_shares], axis=0)
        return combined


def quantize_to_field(
    data: np.ndarray, 
    prime: int, 
    scale: int = 16
) -> np.ndarray:
    """
    Quantize real-valued data to finite field elements.
    
    Args:
        data: Real-valued numpy array
        prime: Prime modulus for the field
        scale: Number of bits for scaling (precision)
        
    Returns:
        Quantized data as integers in Z_p
    """
    scaled = data * (2 ** scale)
    quantized = np.round(scaled).astype(np.int64)
    
    # Map negative values to field
    result = quantized % prime
    return result


def dequantize_from_field(
    data: np.ndarray, 
    prime: int, 
    scale: int = 16
) -> np.ndarray:
    """
    Convert field elements back to real values.
    
    Args:
        data: Integer data in Z_p
        prime: Prime modulus
        scale: Scaling factor used in quantization
        
    Returns:
        Real-valued numpy array
    """
    # Handle wrap-around for negative numbers
    half_prime = prime // 2
    result = np.where(data > half_prime, data.astype(np.float64) - prime, data.astype(np.float64))
    result = result / (2 ** scale)
    return result


if __name__ == '__main__':
    # Basic functionality test
    print("Testing Shamir Secret Sharing...")
    
    prime = 2**27 + 1
    ss = ShamirSecretSharing(threshold=3, num_shares=10, prime=prime)
    
    # Test secret sharing
    secret = 12345
    shares = ss.share_secret(secret)
    reconstructed = ss.reconstruct_secret(shares[:3])
    assert reconstructed == secret, "Secret reconstruction failed"
    
    # Test with different subset of shares
    reconstructed2 = ss.reconstruct_secret(shares[2:5])
    assert reconstructed2 == secret, "Reconstruction with different shares failed"
    
    # Test matrix sharing
    matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    share_matrices = ss.share_matrix(matrix)
    reconstructed_matrix = ss.reconstruct_matrix(share_matrices[:3])
    assert np.array_equal(matrix, reconstructed_matrix), "Matrix reconstruction failed"
    
    # Test homomorphic addition
    secret1, secret2 = 100, 200
    shares1 = ss.share_secret(secret1)
    shares2 = ss.share_secret(secret2)
    added_shares = [(s1[0], ss.add_shares(s1[1], s2[1])) 
                   for s1, s2 in zip(shares1, shares2)]
    reconstructed_sum = ss.reconstruct_secret(added_shares[:3])
    assert reconstructed_sum == secret1 + secret2, "Homomorphic addition failed"
    
    # Test quantization
    real_data = np.array([[0.5, -0.3], [1.2, -0.8]])
    quantized = quantize_to_field(real_data, prime, scale=16)
    dequantized = dequantize_from_field(quantized, prime, scale=16)
    assert np.allclose(real_data, dequantized, atol=1e-4), "Quantization roundtrip failed"
    
    print("All tests passed!")
