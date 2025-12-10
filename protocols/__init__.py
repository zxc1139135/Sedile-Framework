"""
Privacy-preserving protocols for Sedile framework.

Implements three core protocols:
1. Similarity-driven client partitioning
2. Intra-partition data sharing
3. Inter-partition model training
"""

from .partitioning import (
    SimilarityMetric,
    PrivacyPreservingSimilarity,
    ClientPartitioner,
    create_partitioner
)

from .data_sharing import (
    ShareConfig,
    EncodedDataset,
    DataQuantizer,
    IntraPartitionSharing,
    PartitionDataManager
)

from .training import (
    GradientUpdate,
    AggregatedGradient,
    LocalTrainer,
    SecureAggregator,
    FederatedTrainer,
    DistributedTrainingOrchestrator
)

__all__ = [
    # Partitioning
    'SimilarityMetric',
    'PrivacyPreservingSimilarity',
    'ClientPartitioner',
    'create_partitioner',
    # Data sharing
    'ShareConfig',
    'EncodedDataset',
    'DataQuantizer',
    'IntraPartitionSharing',
    'PartitionDataManager',
    # Training
    'GradientUpdate',
    'AggregatedGradient',
    'LocalTrainer',
    'SecureAggregator',
    'FederatedTrainer',
    'DistributedTrainingOrchestrator'
]
