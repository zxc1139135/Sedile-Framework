"""
Harmonic Coding Implementation.

Implements harmonic coding for privacy-preserving gradient computation
in distributed deep learning. Key properties:
- Supports exact computation of non-linear functions (e.g., ReLU)
- Provides privacy protection for gradient updates
- Enables efficient aggregation at the server

Reference: Yu & Avestimehr (2019). Harmonic coding: An optimal linear code 
for privacy-preserving gradient-type computation.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import random


class HarmonicCoder:
    """
    Harmonic coding for privacy-preserving gradient computation.
    
    Encodes data partitions to enable exact gradient computation while
    protecting individual client data from the server.
    """
    
    def __init__(
        self, 
        num_partitions: int, 
        degree: int, 
        prime: int,
        coding_param_c: int = 100
    ):
        """
        Initialize the harmonic coder.
        
        Args:
            num_partitions: K - number of data partitions
            degree: d - degree of gradient function
            prime: Prime modulus for field operations
            coding_param_c: Parameter c for encoding
        """
        self.K = num_partitions
        self.d = degree
        self.prime = prime
        self.c = coding_param_c
        
        # Generate encoding parameters
        self._generate_parameters()
    
    def _generate_parameters(self):
        """Generate encoding parameters beta_1, ..., beta_{d-1}."""
        # Forbidden values for beta
        forbidden = {0}
        for k in range(self.K + 1):
            if self.c - k != 0:
                forbidden.add((self.c * pow(self.c - k, -1, self.prime)) % self.prime)
        
        # Select distinct beta values
        self.betas = []
        candidate = 1
        while len(self.betas) < self.d - 1:
            if candidate not in forbidden and candidate not in self.betas:
                self.betas.append(candidate)
            candidate += 1
    
    def _compute_intermediate_P(
        self, 
        data_partitions: List[np.ndarray], 
        mask_Z: np.ndarray
    ) -> List[np.ndarray]:
        """
        Compute intermediate masking variables P_0, ..., P_K.
        
        P_k = (c / (c-k)) * Z - (1 / (c-k)) * sum_{a=1}^{k} X_a
        """
        P = []
        cumsum = np.zeros_like(mask_Z)
        
        for k in range(self.K + 1):
            if k > 0 and k <= len(data_partitions):
                cumsum = cumsum + data_partitions[k - 1]
            
            c_minus_k = self.c - k
            if c_minus_k == 0:
                raise ValueError(f"Invalid c-k=0 for k={k}")
            
            c_minus_k_inv = pow(c_minus_k, self.prime - 2, self.prime)
            c_term = (self.c * c_minus_k_inv) % self.prime
            
            P_k = (c_term * mask_Z - c_minus_k_inv * cumsum) % self.prime
            P.append(P_k.astype(np.int64))
        
        return P
    
    def encode(
        self, 
        data: np.ndarray, 
        group_id: int, 
        client_idx_in_group: int,
        mask_Z: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode data partition using harmonic coding.
        
        Args:
            data: Data partition to encode (already a secret share)
            group_id: Which group this client belongs to (1 to K, or K+1, K+2)
            client_idx_in_group: Index within the group (0 to d-2)
            mask_Z: Random masking matrix (generated if None)
            
        Returns:
            Tuple of (encoded_data, mask_Z)
        """
        if mask_Z is None:
            mask_Z = np.random.randint(0, self.prime, size=data.shape, dtype=np.int64)
        
        # For special clients j1, j2 (groups K+1, K+2)
        if group_id == self.K + 1:
            # Return P_0
            P = self._compute_intermediate_P([data], mask_Z)
            return P[0], mask_Z
        elif group_id == self.K + 2:
            # Return P_K (need all partitions, simplified here)
            P = self._compute_intermediate_P([data] * self.K, mask_Z)
            return P[-1], mask_Z
        
        # For regular clients in groups 1 to K
        k = group_id  # Group index corresponds to partition index
        beta_idx = client_idx_in_group
        
        if beta_idx >= len(self.betas):
            raise ValueError(f"Client index {beta_idx} exceeds available betas")
        
        beta = self.betas[beta_idx]
        
        # Compute P_{k-1}
        P = self._compute_intermediate_P([data] * k, mask_Z)
        P_k_minus_1 = P[k - 1] if k > 0 else mask_Z
        
        # G_k(X) = X_k + (P_{k-1} - X_k) * beta * (c-k+1) / c
        c_minus_k_plus_1 = self.c - k + 1
        c_inv = pow(self.c, self.prime - 2, self.prime)
        
        factor = (beta * c_minus_k_plus_1 * c_inv) % self.prime
        encoded = (data + (P_k_minus_1 - data) * factor) % self.prime
        
        return encoded.astype(np.int64), mask_Z
    
    def compute_decoding_coefficients(self, k: int) -> np.ndarray:
        """
        Compute Lagrange interpolation coefficients for decoding.
        
        Args:
            k: Partition index
            
        Returns:
            Array of decoding coefficients
        """
        coeffs = np.zeros(self.d - 1, dtype=np.float64)
        c = self.c
        
        for j in range(self.d - 1):
            beta_j = self.betas[j]
            
            # Compute product term
            prod = 1.0
            for j_prime in range(self.d - 1):
                if j_prime != j:
                    beta_j_prime = self.betas[j_prime]
                    if beta_j_prime != beta_j:
                        prod *= beta_j_prime / (beta_j_prime - beta_j)
            
            # Compute denominator terms
            denom1 = c - beta_j * (c - k + 1)
            denom2 = c - beta_j * (c - k)
            
            if denom1 != 0 and denom2 != 0:
                coeffs[j] = (c * c * prod) / (denom1 * denom2)
        
        return coeffs


class HarmonicEncoder:
    """
    Complete harmonic encoding workflow for a client partition.
    """
    
    def __init__(
        self, 
        num_clients_in_partition: int,
        num_data_partitions: int,
        gradient_degree: int,
        prime: int
    ):
        """
        Initialize encoder for a partition.
        
        Args:
            num_clients_in_partition: N_l - clients in this partition
            num_data_partitions: K - data partitions for encoding
            gradient_degree: d - degree of gradient function
            prime: Prime modulus
        """
        self.N_l = num_clients_in_partition
        self.K = num_data_partitions
        self.d = gradient_degree
        self.prime = prime
        
        # Ensure we have enough clients
        required_clients = self.K * (self.d - 1) + 2
        if self.N_l < required_clients:
            # Adjust K to fit available clients
            self.K = max(1, (self.N_l - 2) // (self.d - 1))
        
        self.coder = HarmonicCoder(self.K, self.d, prime)
        
        # Assign clients to groups
        self._assign_clients_to_groups()
    
    def _assign_clients_to_groups(self):
        """Assign clients to harmonic coding groups."""
        self.client_groups = {}
        client_idx = 0
        
        # First K groups with (d-1) clients each
        for group in range(1, self.K + 1):
            for idx_in_group in range(self.d - 1):
                if client_idx < self.N_l - 2:
                    self.client_groups[client_idx] = (group, idx_in_group)
                    client_idx += 1
        
        # Special clients j1 and j2
        if client_idx < self.N_l:
            self.client_groups[client_idx] = (self.K + 1, 0)  # j1
            client_idx += 1
        if client_idx < self.N_l:
            self.client_groups[client_idx] = (self.K + 2, 0)  # j2
    
    def get_client_assignment(self, client_id: int) -> Tuple[int, int]:
        """Get group assignment for a client."""
        if client_id in self.client_groups:
            return self.client_groups[client_id]
        # Default assignment for extra clients
        return (1, 0)
    
    def partition_data(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Partition data into K roughly equal parts.
        
        Args:
            data: Combined data to partition
            
        Returns:
            List of K data partitions
        """
        return np.array_split(data, self.K, axis=0)
    
    def encode_for_client(
        self, 
        client_id: int, 
        data_partition: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate encoded data for a specific client.
        
        Args:
            client_id: Target client ID
            data_partition: Relevant data partition
            mask: Random mask (generated if None)
            
        Returns:
            Tuple of (encoded_data, mask)
        """
        group_id, idx_in_group = self.get_client_assignment(client_id)
        return self.coder.encode(data_partition, group_id, idx_in_group, mask)


class GradientDecoder:
    """
    Decode aggregated gradients using harmonic coding properties.
    """
    
    def __init__(self, harmonic_coder: HarmonicCoder):
        """
        Initialize decoder.
        
        Args:
            harmonic_coder: The coder used for encoding
        """
        self.coder = harmonic_coder
    
    def decode_gradient(
        self, 
        encoded_gradients: Dict[int, np.ndarray],
        boundary_gradients: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        Decode aggregated gradient from encoded components.
        
        Args:
            encoded_gradients: Dict mapping group_id to encoded gradient
            boundary_gradients: (g(P_0), g(P_K)) boundary terms
            
        Returns:
            Decoded global gradient
        """
        K = self.coder.K
        
        # Compute U_k values for each partition
        U_values = []
        for k in range(1, K + 1):
            coeffs = self.coder.compute_decoding_coefficients(k)
            
            # Linear combination of encoded gradients in group k
            if k in encoded_gradients:
                U_k = np.zeros_like(encoded_gradients[k])
                for j, coeff in enumerate(coeffs):
                    if k in encoded_gradients:
                        U_k += coeff * encoded_gradients[k]
                U_values.append(U_k)
        
        # Sum U values and remove boundary terms
        if U_values:
            gradient_sum = sum(U_values)
            g_P0, g_PK = boundary_gradients
            decoded = gradient_sum - g_P0 - g_PK
            return decoded
        
        return np.zeros(1)


def create_harmonic_system(
    num_clients: int,
    num_partitions: int = 5,
    prime: int = 2**27 + 1
) -> HarmonicEncoder:
    """
    Factory function to create harmonic encoding system.
    
    Args:
        num_clients: Number of clients in partition
        num_partitions: K parameter
        prime: Field prime
        
    Returns:
        Configured HarmonicEncoder
    """
    # Gradient degree for typical neural networks
    gradient_degree = 3  # Suitable for most activation functions
    
    return HarmonicEncoder(
        num_clients_in_partition=num_clients,
        num_data_partitions=num_partitions,
        gradient_degree=gradient_degree,
        prime=prime
    )


if __name__ == '__main__':
    # Basic functionality test
    print("Testing Harmonic Coding...")
    
    prime = 2**27 + 1
    
    # Test HarmonicCoder
    coder = HarmonicCoder(num_partitions=3, degree=3, prime=prime)
    
    # Create test data
    data = np.random.randint(0, 1000, size=(10, 5), dtype=np.int64)
    mask = np.random.randint(0, prime, size=data.shape, dtype=np.int64)
    
    # Test encoding for different groups
    encoded1, _ = coder.encode(data, group_id=1, client_idx_in_group=0, mask_Z=mask)
    encoded2, _ = coder.encode(data, group_id=2, client_idx_in_group=1, mask_Z=mask)
    
    assert encoded1.shape == data.shape, "Encoding shape mismatch"
    assert encoded2.shape == data.shape, "Encoding shape mismatch"
    
    # Test HarmonicEncoder
    encoder = create_harmonic_system(num_clients=10, num_partitions=3)
    
    # Test data partitioning
    full_data = np.random.randint(0, 1000, size=(30, 5), dtype=np.int64)
    partitions = encoder.partition_data(full_data)
    assert len(partitions) == encoder.K, "Partition count mismatch"
    
    # Test client assignment
    for client_id in range(10):
        group, idx = encoder.get_client_assignment(client_id)
        assert 1 <= group <= encoder.K + 2, f"Invalid group {group}"
    
    print("All tests passed!")
