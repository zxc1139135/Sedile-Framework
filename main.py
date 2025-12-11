"""
Main experiment runner for Sedile framework.

Executes the complete privacy-preserving distributed learning pipeline:
1. Data loading and non-IID partitioning
2. Similarity-driven client partitioning
3. Intra-partition data sharing
4. Inter-partition model training
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, ExperimentConfig
from data import FederatedDataLoader
from models import get_model, count_parameters
from protocols import (
    create_partitioner,
    PartitionDataManager,
    DistributedTrainingOrchestrator
)
from utils import (
    set_seed,
    setup_logging,
    MetricsTracker,
    ResultsSaver,
    plot_training_curves,
    plot_distribution_heatmap,
    get_device,
    format_time
)


class SedileExperiment:
    """
    Main experiment class for Sedile framework.
    
    Orchestrates the complete training pipeline with privacy-preserving
    protocols for distributed deep learning.
    """
    
    def __init__(
        self,
        dataset: str = 'mnist',
        num_clients: int = 50,
        num_partitions: int = 5,
        distribution_type: str = 'dirichlet',
        distribution_param: float = 0.1,
        num_rounds: int = 300,
        local_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 0.015,
        similarity_metric: str = 'euclidean',
        threshold_t: int = 3,
        seed: int = 42,
        output_dir: str = './outputs',
        device: Optional[str] = None
    ):
        """
        Initialize experiment.
        
        Args:
            dataset: Dataset name ('mnist', 'fmnist', 'cifar10', 'svhn')
            num_clients: Total number of clients N
            num_partitions: Number of partitions V
            distribution_type: 'dirichlet' or 'pathological'
            distribution_param: alpha for dirichlet, kappa for pathological
            num_rounds: Number of training rounds
            local_epochs: Local epochs per round
            batch_size: Training batch size
            learning_rate: Learning rate
            similarity_metric: 'euclidean', 'cosine', or 'kl'
            threshold_t: T-privacy threshold
            seed: Random seed
            output_dir: Directory for outputs
            device: Training device
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.num_partitions = num_partitions
        self.distribution_type = distribution_type
        self.distribution_param = distribution_param
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.similarity_metric = similarity_metric
        self.threshold_t = threshold_t
        self.seed = seed
        self.device = device if device else get_device()
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(
            output_dir,
            f'{dataset}_{distribution_type}_{distribution_param}_{timestamp}'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.logger = setup_logging(
            os.path.join(self.output_dir, 'logs'),
            experiment_name=f'sedile_{dataset}'
        )
        self.results_saver = ResultsSaver(self.output_dir)
        self.metrics_tracker = MetricsTracker()
        
        # Save configuration
        self._save_config()
    
    def _save_config(self):
        """Save experiment configuration."""
        config = {
            'dataset': self.dataset,
            'num_clients': self.num_clients,
            'num_partitions': self.num_partitions,
            'distribution_type': self.distribution_type,
            'distribution_param': self.distribution_param,
            'num_rounds': self.num_rounds,
            'local_epochs': self.local_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'similarity_metric': self.similarity_metric,
            'threshold_t': self.threshold_t,
            'seed': self.seed,
            'device': self.device
        }
        self.results_saver.save_config(config)
    
    def run(self) -> Dict[str, List[float]]:
        """
        Execute complete experiment.
        
        Returns:
            Training history dictionary
        """
        set_seed(self.seed)
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Sedile Experiment")
        self.logger.info("=" * 60)
        self.logger.info(f"Dataset: {self.dataset}")
        self.logger.info(f"Clients: {self.num_clients}, Partitions: {self.num_partitions}")
        self.logger.info(f"Distribution: {self.distribution_type}({self.distribution_param})")
        self.logger.info(f"Device: {self.device}")
        
        # Phase 1: Data Loading and Non-IID Distribution
        self.logger.info("\n[Phase 1] Loading data and creating non-IID distribution...")
        data_loader = self._load_data()
        
        # Phase 2: Similarity-Driven Client Partitioning
        self.logger.info("\n[Phase 2] Executing client partitioning protocol...")
        partitions = self._partition_clients(data_loader)
        
        # Phase 3: Model Initialization
        self.logger.info("\n[Phase 3] Initializing model...")
        model = self._initialize_model()
        
        # Phase 4: Training
        self.logger.info("\n[Phase 4] Starting distributed training...")
        history = self._train(model, data_loader, partitions)
        
        # Save results
        self._save_results(history, model, data_loader, partitions)
        
        elapsed = time.time() - start_time
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"Experiment completed in {format_time(elapsed)}")
        self.logger.info(f"Final test accuracy: {history['test_accuracy'][-1]:.2f}%")
        self.logger.info("=" * 60)
        
        return history
    
    def _load_data(self) -> FederatedDataLoader:
        """Load and partition data among clients."""
        data_loader = FederatedDataLoader(
            dataset_name=self.dataset,
            num_clients=self.num_clients,
            distribution_type=self.distribution_type,
            distribution_param=self.distribution_param,
            batch_size=self.batch_size,
            data_dir='./data/raw',
            seed=self.seed
        )
        
        self.logger.info(f"  Total training samples: {sum(data_loader.get_client_data_size(i) for i in range(self.num_clients))}")
        self.logger.info(f"  Average samples per client: {np.mean([data_loader.get_client_data_size(i) for i in range(self.num_clients)]):.1f}")
        
        return data_loader
    
    def _partition_clients(
        self, 
        data_loader: FederatedDataLoader
    ) -> List[List[int]]:
        """Execute similarity-driven client partitioning."""
        distributions = data_loader.get_all_distributions()
        
        partitioner = create_partitioner(
            num_clients=self.num_clients,
            num_partitions=self.num_partitions,
            metric=self.similarity_metric,
            use_encryption=False  # Set to True for full privacy
        )
        
        partitions = partitioner.partition(distributions, seed=self.seed)
        
        # Evaluate partitioning quality
        metrics = partitioner.evaluate_partitioning(partitions, distributions)
        
        self.logger.info(f"  Partition sizes: {[len(p) for p in partitions]}")
        self.logger.info(f"  Inter-partition variance: {metrics['inter_partition_variance']:.6f}")
        self.logger.info(f"  Avg intra-partition variance: {metrics['avg_intra_partition_variance']:.6f}")
        
        return partitions
    
    def _initialize_model(self) -> torch.nn.Module:
        """Initialize the neural network model."""
        model = get_model(self.dataset, num_classes=10)
        model = model.to(self.device)
        
        num_params = count_parameters(model)
        self.logger.info(f"  Model: {type(model).__name__}")
        self.logger.info(f"  Parameters: {num_params:,}")
        
        return model
    
    def _train(
        self,
        model: torch.nn.Module,
        data_loader: FederatedDataLoader,
        partitions: List[List[int]]
    ) -> Dict[str, List[float]]:
        """Execute distributed training."""
        # Create client data loaders
        client_data_loaders = {
            i: data_loader.get_client_loader(i)
            for i in range(self.num_clients)
        }
        
        test_loader = data_loader.get_test_loader()
        
        # Initialize training orchestrator
        orchestrator = DistributedTrainingOrchestrator(
            model=model,
            partitions=partitions,
            learning_rate=self.learning_rate,
            local_epochs=self.local_epochs,
            device=self.device
        )
        
        # Training loop with logging
        history = orchestrator.train(
            client_data_loaders=client_data_loaders,
            test_loader=test_loader,
            num_rounds=self.num_rounds,
            log_interval=10
        )
        
        return history
    
    def _save_results(
        self,
        history: Dict[str, List[float]],
        model: torch.nn.Module,
        data_loader: FederatedDataLoader,
        partitions: List[List[int]]
    ):
        """Save all experiment results."""
        # Save metrics
        self.results_saver.save_metrics(history, 'training_history')
        
        # Save model
        self.results_saver.save_model(model, 'final_model')
        
        # Save partitions
        with open(os.path.join(self.output_dir, 'partitions.json'), 'w') as f:
            json.dump(partitions, f)
        
        # Save distributions
        distributions = data_loader.get_all_distributions()
        self.results_saver.save_numpy(distributions, 'client_distributions')
        
        # Generate plots
        plot_training_curves(
            history,
            save_path=os.path.join(self.output_dir, 'training_curves.png'),
            title=f'Sedile - {self.dataset.upper()} ({self.distribution_type}, {self.distribution_param})'
        )
        
        plot_distribution_heatmap(
            distributions,
            save_path=os.path.join(self.output_dir, 'distribution_heatmap.png'),
            title=f'Client Data Distribution ({self.distribution_type}, {self.distribution_param})'
        )
        
        self.logger.info(f"\nResults saved to: {self.output_dir}")


def run_experiment(args):
    """Run a single experiment with given arguments."""
    experiment = SedileExperiment(
        dataset=args.dataset,
        num_clients=args.num_clients,
        num_partitions=args.num_partitions,
        distribution_type=args.distribution,
        distribution_param=args.param,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        similarity_metric=args.metric,
        threshold_t=args.threshold,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device
    )
    
    history = experiment.run()
    return history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Sedile: Privacy-Preserving Distributed Deep Learning'
    )
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fmnist', 'cifar10', 'svhn'],
                       help='Dataset to use')
    
    # Client settings
    parser.add_argument('--num-clients', type=int, default=50,
                       help='Number of clients')
    parser.add_argument('--num-partitions', type=int, default=5,
                       help='Number of partitions')
    
    # Distribution settings
    parser.add_argument('--distribution', type=str, default='dirichlet',
                       choices=['dirichlet', 'pathological'],
                       help='Data distribution type')
    parser.add_argument('--param', type=float, default=0.1,
                       help='Distribution parameter (alpha for dirichlet, kappa for pathological)')
    
    # Training settings
    parser.add_argument('--rounds', type=int, default=300,
                       help='Number of training rounds')
    parser.add_argument('--local-epochs', type=int, default=3,
                       help='Local epochs per round')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.015,
                       help='Learning rate')
    
    # Protocol settings
    parser.add_argument('--metric', type=str, default='euclidean',
                       choices=['euclidean', 'cosine', 'kl'],
                       help='Similarity metric for partitioning')
    parser.add_argument('--threshold', type=int, default=3,
                       help='T-privacy threshold')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    run_experiment(args)


if __name__ == '__main__':
    main()
