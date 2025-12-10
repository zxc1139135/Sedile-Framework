"""
Intra-Partition Data Sharing Protocol.

Implements the data sharing mechanism within partitions:
1. Dataset quantization to finite field
2. Shamir secret sharing for dataset distribution
3. Harmonic coding for encoded dataset generation

This protocol enables clients to obtain privacy-preserving versions
of the partition's dataset while maintaining T-privacy against
colluding adversaries.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.shamir import (
    ShamirSecretSharing,
    SecretShareDistributor,
    quantize_to_field,
    dequantize_from_field
)
from crypto.harmonic import (
    HarmonicEncoder,
    create_harmonic_system
)


@dataclass
class ShareConfig:
    """Configuration for secret sharing."""
    threshold: int  # T - minimum shares for reconstruction
    prime: int  # Prime modulus
    quantization_scale: int  # Bits for quantization
    

@dataclass
class EncodedDataset:
    """Container for an encoded dataset."""
    data: np.ndarray  # Encoded data
    labels: np.ndarray  # Encoded labels (or original if not encoded)
    mask: Optional[np.ndarray] = None  # Random mask used for encoding
    partition_id: int = 0
    client_id: int = 0


class DataQuantizer:
    """
    Quantize real-valued data to finite field elements.
    
    Transforms floating-point tensors to integers in Z_p
    for cryptographic operations.
    """
    
    def __init__(self, prime: int, scale: int = 16):
        """
        Initialize quantizer.
        
        Args:
            prime: Prime modulus for the field
            scale: Number of bits for scaling
        """
        self.prime = prime
        self.scale = scale
        self.scale_factor = 2 ** scale
    
    def quantize(self, data: torch.Tensor) -> np.ndarray:
        """
        Quantize tensor to field elements.
        
        Args:
            data: Real-valued tensor
            
        Returns:
            Integer array in Z_p
        """
        np_data = data.numpy() if isinstance(data, torch.Tensor) else data
        return quantize_to_field(np_data, self.prime, self.scale)
    
    def dequantize(self, data: np.ndarray) -> torch.Tensor:
        """
        Convert field elements back to real values.
        
        Args:
            data: Integer array in Z_p
            
        Returns:
            Real-valued tensor
        """
        real_data = dequantize_from_field(data, self.prime, self.scale)
        return torch.from_numpy(real_data).float()


class IntraPartitionSharing:
    """
    Manages secret sharing of datasets within a partition.
    
    Each client:
    1. Quantizes its local dataset
    2. Generates Shamir shares for all partition members
    3. Receives shares from other members
    4. Encodes combined shares using harmonic coding
    """
    
    def __init__(
        self,
        partition_id: int,
        client_ids: List[int],
        threshold: int,
        prime: int = 2**27 + 1,
        quantization_scale: int = 16,
        num_data_partitions: int = 5
    ):
        """
        Initialize sharing protocol for a partition.
        
        Args:
            partition_id: ID of this partition
            client_ids: List of client IDs in this partition
            threshold: T-privacy threshold
            prime: Prime modulus
            quantization_scale: Quantization bits
            num_data_partitions: K for harmonic coding
        """
        self.partition_id = partition_id
        self.client_ids = client_ids
        self.num_clients = len(client_ids)
        self.threshold = min(threshold, self.num_clients)
        self.prime = prime
        
        self.quantizer = DataQuantizer(prime, quantization_scale)
        self.ss = ShamirSecretSharing(self.threshold, self.num_clients, prime)
        
        # Harmonic encoder
        self.harmonic_encoder = create_harmonic_system(
            num_clients=self.num_clients,
            num_partitions=num_data_partitions,
            prime=prime
        )
        
        # Storage for shares
        self.local_shares: Dict[int, Dict[int, np.ndarray]] = {}  # client_id -> {sender_id: share}
        self.encoded_datasets: Dict[int, EncodedDataset] = {}
    
    def prepare_local_dataset(
        self,
        client_id: int,
        data: torch.Tensor,
        labels: torch.Tensor
    ) -> np.ndarray:
        """
        Prepare and quantize a client's local dataset.
        
        Args:
            client_id: ID of the client
            data: Client's data tensor
            labels: Client's label tensor
            
        Returns:
            Quantized dataset
        """
        # Flatten data for sharing
        flat_data = data.view(data.size(0), -1)
        quantized = self.quantizer.quantize(flat_data)
        return quantized
    
    def generate_shares(
        self,
        client_id: int,
        quantized_data: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Generate secret shares of client's data for all partition members.
        
        Args:
            client_id: ID of sharing client
            quantized_data: Quantized local dataset
            
        Returns:
            Dict mapping recipient_id to share
        """
        share_matrices = self.ss.share_matrix(quantized_data)
        
        shares = {}
        for idx, recipient_id in enumerate(self.client_ids):
            shares[recipient_id] = share_matrices[idx]
        
        return shares
    
    def receive_share(
        self,
        recipient_id: int,
        sender_id: int,
        share: np.ndarray
    ):
        """
        Store a received share.
        
        Args:
            recipient_id: ID of receiving client
            sender_id: ID of sending client
            share: The secret share
        """
        if recipient_id not in self.local_shares:
            self.local_shares[recipient_id] = {}
        self.local_shares[recipient_id][sender_id] = share
    
    def combine_shares(
        self,
        client_id: int
    ) -> np.ndarray:
        """
        Combine all received shares for a client.
        
        Args:
            client_id: ID of the client
            
        Returns:
            Combined share matrix representing partition data
        """
        if client_id not in self.local_shares:
            raise ValueError(f"No shares received for client {client_id}")
        
        shares = self.local_shares[client_id]
        
        # Sort by sender_id and concatenate
        sorted_shares = [shares[sid] for sid in sorted(shares.keys())]
        combined = np.concatenate(sorted_shares, axis=0)
        
        return combined
    
    def encode_dataset(
        self,
        client_id: int,
        combined_shares: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> EncodedDataset:
        """
        Apply harmonic coding to combined shares.
        
        Args:
            client_id: Client ID
            combined_shares: Combined secret shares
            labels: Optional labels
            
        Returns:
            EncodedDataset with harmonic-coded data
        """
        # Get client's group assignment
        client_idx = self.client_ids.index(client_id)
        group_id, idx_in_group = self.harmonic_encoder.get_client_assignment(client_idx)
        
        # Partition data for encoding
        partitions = self.harmonic_encoder.partition_data(combined_shares)
        
        # Select relevant partition based on group
        partition_idx = min(group_id - 1, len(partitions) - 1)
        data_partition = partitions[partition_idx]
        
        # Apply harmonic encoding
        encoded_data, mask = self.harmonic_encoder.encode_for_client(
            client_idx,
            data_partition.astype(np.int64)
        )
        
        encoded = EncodedDataset(
            data=encoded_data,
            labels=labels if labels is not None else np.array([]),
            mask=mask,
            partition_id=self.partition_id,
            client_id=client_id
        )
        
        self.encoded_datasets[client_id] = encoded
        return encoded
    
    def execute_sharing_protocol(
        self,
        client_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[int, EncodedDataset]:
        """
        Execute complete sharing protocol.
        
        Args:
            client_data: Dict mapping client_id to (data, labels) tuples
            
        Returns:
            Dict mapping client_id to EncodedDataset
        """
        # Phase 1: Quantize and generate shares
        all_shares: Dict[int, Dict[int, np.ndarray]] = {}
        
        for client_id in self.client_ids:
            if client_id in client_data:
                data, labels = client_data[client_id]
                quantized = self.prepare_local_dataset(client_id, data, labels)
                shares = self.generate_shares(client_id, quantized)
                all_shares[client_id] = shares
        
        # Phase 2: Distribute shares
        for sender_id, shares in all_shares.items():
            for recipient_id, share in shares.items():
                self.receive_share(recipient_id, sender_id, share)
        
        # Phase 3: Combine and encode
        results = {}
        for client_id in self.client_ids:
            if client_id in self.local_shares:
                combined = self.combine_shares(client_id)
                
                # Get labels
                labels = None
                if client_id in client_data:
                    _, labels_tensor = client_data[client_id]
                    labels = labels_tensor.numpy()
                
                encoded = self.encode_dataset(client_id, combined, labels)
                results[client_id] = encoded
        
        return results


class PartitionDataManager:
    """
    High-level manager for partition data sharing.
    """
    
    def __init__(
        self,
        partitions: List[List[int]],
        threshold: int = 3,
        prime: int = 2**27 + 1
    ):
        """
        Initialize data manager.
        
        Args:
            partitions: List of client ID lists per partition
            threshold: T-privacy threshold
            prime: Prime modulus
        """
        self.partitions = partitions
        self.num_partitions = len(partitions)
        self.threshold = threshold
        self.prime = prime
        
        # Create sharing protocol for each partition
        self.sharing_protocols: Dict[int, IntraPartitionSharing] = {}
        for partition_id, client_ids in enumerate(partitions):
            self.sharing_protocols[partition_id] = IntraPartitionSharing(
                partition_id=partition_id,
                client_ids=client_ids,
                threshold=threshold,
                prime=prime
            )
    
    def get_partition_for_client(self, client_id: int) -> int:
        """Get partition ID for a client."""
        for partition_id, client_ids in enumerate(self.partitions):
            if client_id in client_ids:
                return partition_id
        raise ValueError(f"Client {client_id} not found in any partition")
    
    def share_data_in_partition(
        self,
        partition_id: int,
        client_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[int, EncodedDataset]:
        """
        Execute data sharing for a specific partition.
        
        Args:
            partition_id: ID of the partition
            client_data: Client data for this partition
            
        Returns:
            Encoded datasets for all clients in partition
        """
        if partition_id not in self.sharing_protocols:
            raise ValueError(f"Invalid partition ID: {partition_id}")
        
        return self.sharing_protocols[partition_id].execute_sharing_protocol(client_data)
    
    def share_all_data(
        self,
        all_client_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[int, EncodedDataset]:
        """
        Execute data sharing for all partitions.
        
        Args:
            all_client_data: Data for all clients
            
        Returns:
            Encoded datasets for all clients
        """
        all_encoded = {}
        
        for partition_id, client_ids in enumerate(self.partitions):
            # Filter data for this partition
            partition_data = {
                cid: all_client_data[cid] 
                for cid in client_ids 
                if cid in all_client_data
            }
            
            encoded = self.share_data_in_partition(partition_id, partition_data)
            all_encoded.update(encoded)
        
        return all_encoded


if __name__ == '__main__':
    # Test data sharing protocol
    print("Testing Intra-Partition Data Sharing Protocol...")
    
    # Create test data
    num_clients = 10
    data_size = 100
    feature_dim = 784
    num_classes = 10
    
    # Generate random client data
    client_data = {}
    for i in range(num_clients):
        data = torch.randn(data_size // num_clients, 1, 28, 28)
        labels = torch.randint(0, num_classes, (data_size // num_clients,))
        client_data[i] = (data, labels)
    
    # Create partitions
    partitions = [list(range(5)), list(range(5, 10))]
    
    # Initialize manager
    manager = PartitionDataManager(
        partitions=partitions,
        threshold=3,
        prime=2**27 + 1
    )
    
    # Test sharing for first partition
    partition_data = {cid: client_data[cid] for cid in partitions[0]}
    encoded = manager.share_data_in_partition(0, partition_data)
    
    print(f"Number of encoded datasets: {len(encoded)}")
    for client_id, enc_data in encoded.items():
        print(f"  Client {client_id}: encoded shape = {enc_data.data.shape}")
    
    print("\nAll tests passed!")
