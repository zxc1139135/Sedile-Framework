"""
Privacy-preserving protocols for Sedile framework.
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
