"""
Configuration settings for Sedile framework.
Experimental parameters aligned with paper specifications.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CryptoConfig:
    """Cryptographic parameters configuration."""
    # Paillier encryption security parameter
    paillier_key_size: int = 2048
    # Shamir secret sharing threshold
    threshold_t: int = 3
    # Finite field prime for secret sharing (2^27 + 1 for MNIST compatibility)
    prime_p: int = 2**27 + 1
    # Quantization bit precision
    quantization_bits: int = 16


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Available datasets
    datasets: List[str] = field(default_factory=lambda: [
        'mnist', 'fmnist', 'cifar10', 'svhn'
    ])
    # Data directory
    data_dir: str = './data/raw'
    # Input dimensions per dataset
    input_dims: dict = field(default_factory=lambda: {
        'mnist': (1, 28, 28),
        'fmnist': (1, 28, 28),
        'cifar10': (3, 32, 32),
        'svhn': (3, 32, 32)
    })
    # Number of classes
    num_classes: int = 10


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Model types for each dataset
    model_mapping: dict = field(default_factory=lambda: {
        'mnist': 'mlp',
        'fmnist': 'lenet',
        'cifar10': 'vgg',
        'svhn': 'resnet18'
    })
    # MLP hidden layer sizes
    mlp_hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Number of clients
    num_clients: int = 50
    # Number of partitions
    num_partitions: int = 5
    # Local training rounds per global round
    local_epochs: int = 3
    # Batch size
    batch_size: int = 32
    # Learning rate
    learning_rate: float = 0.015
    # Maximum global rounds
    max_rounds: int = 300
    # Convergence threshold
    convergence_threshold: float = 1e-4


@dataclass
class DistributionConfig:
    """Non-IID data distribution configuration."""
    # Distribution types
    dirichlet_alphas: List[float] = field(default_factory=lambda: [0.1, 1.0])
    pathological_kappas: List[int] = field(default_factory=lambda: [2, 5])
    # Default distribution type ('dirichlet' or 'pathological')
    default_type: str = 'dirichlet'
    # Default parameter
    default_alpha: float = 0.1
    default_kappa: int = 2


@dataclass
class HarmonicConfig:
    """Harmonic coding configuration."""
    # Number of data partitions K
    num_data_partitions: int = 5
    # Coding parameter c
    coding_param_c: int = 100


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    harmonic: HarmonicConfig = field(default_factory=HarmonicConfig)
    
    # Experiment settings
    seed: int = 42
    device: str = 'cuda'
    log_interval: int = 10
    save_interval: int = 50
    output_dir: str = './outputs'
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()


def get_config(
    dataset: str = 'mnist',
    distribution: str = 'dirichlet',
    param: float = 0.1,
    num_clients: int = 50,
    num_partitions: int = 5
) -> ExperimentConfig:
    """
    Get configuration for specific experiment setup.
    
    Args:
        dataset: Dataset name ('mnist', 'fmnist', 'cifar10', 'svhn')
        distribution: Distribution type ('dirichlet' or 'pathological')
        param: Distribution parameter (alpha for dirichlet, kappa for pathological)
        num_clients: Number of participating clients
        num_partitions: Number of client partitions
        
    Returns:
        Configured ExperimentConfig instance
    """
    config = ExperimentConfig()
    config.training.num_clients = num_clients
    config.training.num_partitions = num_partitions
    config.distribution.default_type = distribution
    
    if distribution == 'dirichlet':
        config.distribution.default_alpha = param
    else:
        config.distribution.default_kappa = int(param)
    
    return config
