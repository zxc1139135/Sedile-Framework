"""
Main Experiment Runner for Sedile Framework.

Orchestrates the complete privacy-preserving distributed learning pipeline:
1. Data loading and non-IID distribution
2. Similarity-driven client partitioning
3. Intra-partition data sharing
4. Inter-partition model training

Experimental configurations match the paper specifications.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, get_config
from data import FederatedDataLoader
from models import get_model
from protocols import (
    create_partitioner,
    PartitionDataManager,
    DistributedTrainingOrchestrator
)
from utils import (
    setup_seed,
    get_device,
    setup_logging,
    MetricTracker,
    Timer,
    save_results,
    compute_model_size,
    compute_communication_cost,
    plot_training_curves,
    create_experiment_report
)


class SedileExperiment:
    """
    Main experiment class for Sedile framework.
    """
    
    def __init__(
        self,
        dataset: str = 'mnist',
        distribution: str = 'dirichlet',
        distribution_param: float = 0.1,
        num_clients: int = 50,
        num_partitions: int = 5,
        threshold: int = 3,
        num_rounds: int = 300,
        local_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 0.015,
        seed: int = 42,
        output_dir: str = './outputs',
        device: Optional[str] = None
    ):
        """
        Initialize experiment.
        
        Args:
            dataset: Dataset name ('mnist', 'fmnist', 'cifar10', 'svhn')
            distribution: Data distribution ('dirichlet', 'pathological')
            distribution_param: Distribution parameter
            num_clients: Number of clients N
            num_partitions: Number of partitions V
            threshold: Privacy threshold T
            num_rounds: Number of training rounds
            local_epochs: Local epochs per round
            batch_size: Training batch size
            learning_rate: Learning rate
            seed: Random seed
            output_dir: Output directory
            device: Training device
        """
        self.dataset = dataset
        self.distribution = distribution
        self.distribution_param = distribution_param
        self.num_clients = num_clients
        self.num_partitions = num_partitions
        self.threshold = threshold
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.output_dir = output_dir
        
        # Setup
        setup_seed(seed)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_name = f"{dataset}_{distribution}_{distribution_param}_{timestamp}"
        self.exp_dir = os.path.join(output_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Components (initialized in setup)
        self.data_loader = None
        self.model = None
        self.partitions = None
        self.trainer = None
        
        # Results
        self.metrics = MetricTracker()
        self.timer = Timer()
    
    def setup(self):
        """Setup experiment components."""
        print(f"Setting up experiment: {self.exp_name}")
        print(f"Device: {self.device}")
        
        # Load data
        print(f"Loading {self.dataset} dataset...")
        self.data_loader = FederatedDataLoader(
            dataset_name=self.dataset,
            num_clients=self.num_clients,
            distribution_type=self.distribution,
            distribution_param=self.distribution_param,
            batch_size=self.batch_size,
            seed=self.seed
        )
        
        # Get client distributions
        distributions = self.data_loader.get_all_distributions()
        
        # Partition clients
        print(f"Partitioning {self.num_clients} clients into {self.num_partitions} partitions...")
        partitioner = create_partitioner(
            num_clients=self.num_clients,
            num_partitions=self.num_partitions,
            metric='euclidean'
        )
        self.partitions = partitioner.partition(distributions, seed=self.seed)
        
        # Print partition sizes
        print(f"Partition sizes: {[len(p) for p in self.partitions]}")
        
        # Create model
        print(f"Creating model for {self.dataset}...")
        self.model = get_model(self.dataset)
        model_size = compute_model_size(self.model)
        print(f"Model parameters: {model_size['total_params']:,}")
        
        # Create trainer
        self.trainer = DistributedTrainingOrchestrator(
            model=self.model,
            partitions=self.partitions,
            learning_rate=self.learning_rate,
            local_epochs=self.local_epochs,
            device=self.device
        )
        
        # Save distributions visualization
        self._save_distributions(distributions)
        
        print("Setup complete!")
    
    def _save_distributions(self, distributions: np.ndarray):
        """Save distribution visualization."""
        from utils.visualization import (
            plot_client_distributions,
            plot_partition_distributions
        )
        
        plot_client_distributions(
            distributions,
            save_path=os.path.join(self.exp_dir, 'client_distributions.png'),
            title=f'{self.dataset} - Client Data Distributions (alpha={self.distribution_param})'
        )
        
        plot_partition_distributions(
            distributions,
            self.partitions,
            save_path=os.path.join(self.exp_dir, 'partition_distributions.png'),
            title=f'{self.dataset} - Partition Data Distributions'
        )
    
    def run(self) -> Dict:
        """
        Run the experiment.
        
        Returns:
            Dictionary with experiment results
        """
        print(f"\nStarting training for {self.num_rounds} rounds...")
        
        # Prepare data loaders
        client_data_loaders = {
            cid: self.data_loader.get_client_loader(cid)
            for cid in range(self.num_clients)
        }
        test_loader = self.data_loader.get_test_loader()
        
        # Training
        self.timer.start()
        history = self.trainer.train(
            client_data_loaders=client_data_loaders,
            test_loader=test_loader,
            num_rounds=self.num_rounds,
            log_interval=10
        )
        training_time = self.timer.stop('training')
        
        # Final evaluation
        final_accuracy = history['test_accuracy'][-1] if history['test_accuracy'] else 0
        best_accuracy = max(history['test_accuracy']) if history['test_accuracy'] else 0
        
        print(f"\nTraining completed in {training_time:.2f}s")
        print(f"Final accuracy: {final_accuracy:.2f}%")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        
        # Save results
        results = self._compile_results(history, training_time)
        self._save_results(results, history)
        
        return results
    
    def _compile_results(
        self, 
        history: Dict[str, List[float]], 
        training_time: float
    ) -> Dict:
        """Compile experiment results."""
        results = {
            'config': {
                'dataset': self.dataset,
                'distribution': self.distribution,
                'distribution_param': self.distribution_param,
                'num_clients': self.num_clients,
                'num_partitions': self.num_partitions,
                'threshold': self.threshold,
                'num_rounds': self.num_rounds,
                'local_epochs': self.local_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'seed': self.seed
            },
            'results': {
                'final_accuracy': history['test_accuracy'][-1] if history['test_accuracy'] else 0,
                'best_accuracy': max(history['test_accuracy']) if history['test_accuracy'] else 0,
                'final_loss': history['test_loss'][-1] if history['test_loss'] else 0,
                'training_time': training_time
            },
            'history': history,
            'partition_sizes': [len(p) for p in self.partitions]
        }
        
        # Add communication cost estimate
        comm_cost = compute_communication_cost(
            self.model, self.num_clients, self.num_rounds
        )
        results['communication'] = comm_cost
        
        return results
    
    def _save_results(self, results: Dict, history: Dict[str, List[float]]):
        """Save results and plots."""
        # Save JSON results
        save_results(results, self.exp_dir, 'results.json')
        
        # Save training curves
        plot_training_curves(
            history,
            save_path=os.path.join(self.exp_dir, 'training_curves.png'),
            title=f'{self.dataset} - Training Progress'
        )
        
        print(f"Results saved to {self.exp_dir}")


def run_single_experiment(args) -> Dict:
    """
    Run a single experiment with given arguments.
    
    Args:
        args: Argument namespace
        
    Returns:
        Experiment results
    """
    experiment = SedileExperiment(
        dataset=args.dataset,
        distribution=args.distribution,
        distribution_param=args.distribution_param,
        num_clients=args.num_clients,
        num_partitions=args.num_partitions,
        threshold=args.threshold,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device
    )
    
    experiment.setup()
    results = experiment.run()
    
    return results


def run_all_experiments(args) -> List[Dict]:
    """
    Run experiments across all datasets and configurations.
    
    Args:
        args: Argument namespace
        
    Returns:
        List of experiment results
    """
    datasets = ['mnist', 'fmnist', 'cifar10', 'svhn']
    distributions = [
        ('dirichlet', 0.1),
        ('dirichlet', 1.0),
        ('pathological', 2),
        ('pathological', 5)
    ]
    
    all_results = []
    
    for dataset in datasets:
        for dist_type, dist_param in distributions:
            print(f"\n{'='*60}")
            print(f"Running: {dataset} with {dist_type} ({dist_param})")
            print(f"{'='*60}")
            
            try:
                experiment = SedileExperiment(
                    dataset=dataset,
                    distribution=dist_type,
                    distribution_param=dist_param,
                    num_clients=args.num_clients,
                    num_partitions=args.num_partitions,
                    threshold=args.threshold,
                    num_rounds=args.num_rounds,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    output_dir=args.output_dir,
                    device=args.device
                )
                
                experiment.setup()
                results = experiment.run()
                all_results.append(results)
                
            except Exception as e:
                print(f"Error running {dataset} with {dist_type}: {e}")
                continue
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'all_experiments_summary.json')
    save_results({'experiments': all_results}, args.output_dir, 'summary.json')
    
    return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Sedile: Privacy-Preserving Distributed Deep Learning'
    )
    
    # Dataset and distribution
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fmnist', 'cifar10', 'svhn'],
                       help='Dataset to use')
    parser.add_argument('--distribution', type=str, default='dirichlet',
                       choices=['dirichlet', 'pathological'],
                       help='Data distribution type')
    parser.add_argument('--distribution-param', type=float, default=0.1,
                       help='Distribution parameter (alpha or kappa)')
    
    # Federated learning settings
    parser.add_argument('--num-clients', type=int, default=50,
                       help='Number of clients')
    parser.add_argument('--num-partitions', type=int, default=5,
                       help='Number of partitions')
    parser.add_argument('--threshold', type=int, default=3,
                       help='Privacy threshold T')
    
    # Training settings
    parser.add_argument('--num-rounds', type=int, default=300,
                       help='Number of training rounds')
    parser.add_argument('--local-epochs', type=int, default=3,
                       help='Local epochs per round')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.015,
                       help='Learning rate')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    
    # Run mode
    parser.add_argument('--run-all', action='store_true',
                       help='Run experiments on all datasets')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("="*60)
    print("Sedile: Privacy-Preserving Distributed Deep Learning")
    print("="*60)
    
    if args.run_all:
        results = run_all_experiments(args)
        print(f"\nCompleted {len(results)} experiments")
    else:
        results = run_single_experiment(args)
    
    print("\nExperiment(s) completed!")


if __name__ == '__main__':
    main()
