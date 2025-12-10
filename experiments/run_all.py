"""
Batch experiment runner for reproducing paper results.

Runs experiments across all datasets and distribution settings
as specified in the paper's experimental evaluation.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import SedileExperiment
from utils import set_seed, format_time


# Experiment configurations from the paper
PAPER_EXPERIMENTS = {
    # Dataset configurations
    'datasets': ['mnist', 'fmnist', 'cifar10', 'svhn'],
    
    # Distribution configurations
    'distributions': {
        'dirichlet': [0.1, 1.0],
        'pathological': [2, 5]
    },
    
    # Training parameters
    'num_clients': 50,
    'num_partitions': 5,
    'num_rounds': 300,
    'local_epochs': 3,
    'batch_size': 32,
    'learning_rate': 0.015,
    
    # Privacy parameters
    'threshold_t': 3,
    'similarity_metric': 'euclidean'
}


def generate_experiment_configs(
    datasets: List[str] = None,
    distributions: Dict[str, List] = None
) -> List[Dict[str, Any]]:
    """
    Generate experiment configurations.
    
    Args:
        datasets: List of datasets (uses paper defaults if None)
        distributions: Distribution settings (uses paper defaults if None)
        
    Returns:
        List of experiment configuration dictionaries
    """
    if datasets is None:
        datasets = PAPER_EXPERIMENTS['datasets']
    
    if distributions is None:
        distributions = PAPER_EXPERIMENTS['distributions']
    
    configs = []
    
    for dataset in datasets:
        for dist_type, params in distributions.items():
            for param in params:
                config = {
                    'dataset': dataset,
                    'num_clients': PAPER_EXPERIMENTS['num_clients'],
                    'num_partitions': PAPER_EXPERIMENTS['num_partitions'],
                    'distribution_type': dist_type,
                    'distribution_param': param,
                    'num_rounds': PAPER_EXPERIMENTS['num_rounds'],
                    'local_epochs': PAPER_EXPERIMENTS['local_epochs'],
                    'batch_size': PAPER_EXPERIMENTS['batch_size'],
                    'learning_rate': PAPER_EXPERIMENTS['learning_rate'],
                    'similarity_metric': PAPER_EXPERIMENTS['similarity_metric'],
                    'threshold_t': PAPER_EXPERIMENTS['threshold_t']
                }
                configs.append(config)
    
    return configs


def run_batch_experiments(
    output_dir: str = './outputs/batch',
    datasets: List[str] = None,
    distributions: Dict[str, List] = None,
    seed: int = 42,
    device: str = None
) -> Dict[str, Any]:
    """
    Run batch experiments.
    
    Args:
        output_dir: Base output directory
        datasets: List of datasets to run
        distributions: Distribution settings
        seed: Random seed
        device: Training device
        
    Returns:
        Dictionary with all experiment results
    """
    set_seed(seed)
    
    configs = generate_experiment_configs(datasets, distributions)
    total_experiments = len(configs)
    
    print("=" * 70)
    print("Sedile Batch Experiment Runner")
    print("=" * 70)
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'experiments': [],
        'summary': {},
        'start_time': datetime.now().isoformat()
    }
    
    total_start = time.time()
    
    for idx, config in enumerate(configs, 1):
        exp_name = f"{config['dataset']}_{config['distribution_type']}_{config['distribution_param']}"
        
        print(f"\n[{idx}/{total_experiments}] Running: {exp_name}")
        print("-" * 50)
        
        exp_start = time.time()
        
        try:
            experiment = SedileExperiment(
                dataset=config['dataset'],
                num_clients=config['num_clients'],
                num_partitions=config['num_partitions'],
                distribution_type=config['distribution_type'],
                distribution_param=config['distribution_param'],
                num_rounds=config['num_rounds'],
                local_epochs=config['local_epochs'],
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                similarity_metric=config['similarity_metric'],
                threshold_t=config['threshold_t'],
                seed=seed,
                output_dir=output_dir,
                device=device
            )
            
            history = experiment.run()
            
            exp_result = {
                'config': config,
                'final_accuracy': history['test_accuracy'][-1],
                'best_accuracy': max(history['test_accuracy']),
                'final_loss': history['test_loss'][-1],
                'status': 'success',
                'duration': time.time() - exp_start
            }
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            exp_result = {
                'config': config,
                'status': 'failed',
                'error': str(e),
                'duration': time.time() - exp_start
            }
        
        results['experiments'].append(exp_result)
        
        # Save intermediate results
        with open(os.path.join(output_dir, 'batch_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Completed in {format_time(exp_result['duration'])}")
        if exp_result['status'] == 'success':
            print(f"Final accuracy: {exp_result['final_accuracy']:.2f}%")
    
    # Generate summary
    successful = [e for e in results['experiments'] if e['status'] == 'success']
    
    results['summary'] = {
        'total_experiments': total_experiments,
        'successful': len(successful),
        'failed': total_experiments - len(successful),
        'total_duration': time.time() - total_start,
        'average_accuracy': sum(e['final_accuracy'] for e in successful) / len(successful) if successful else 0
    }
    
    results['end_time'] = datetime.now().isoformat()
    
    # Save final results
    with open(os.path.join(output_dir, 'batch_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Batch Experiments Completed")
    print("=" * 70)
    print(f"Successful: {results['summary']['successful']}/{total_experiments}")
    print(f"Total time: {format_time(results['summary']['total_duration'])}")
    print(f"Average accuracy: {results['summary']['average_accuracy']:.2f}%")
    print("=" * 70)
    
    # Print results table
    print("\nResults Summary:")
    print("-" * 70)
    print(f"{'Dataset':<10} {'Distribution':<15} {'Param':<8} {'Accuracy':<12} {'Status'}")
    print("-" * 70)
    
    for exp in results['experiments']:
        config = exp['config']
        status = exp['status']
        acc = f"{exp.get('final_accuracy', 0):.2f}%" if status == 'success' else 'N/A'
        print(f"{config['dataset']:<10} {config['distribution_type']:<15} {config['distribution_param']:<8} {acc:<12} {status}")
    
    return results


def main():
    """Main entry point for batch experiments."""
    parser = argparse.ArgumentParser(
        description='Sedile Batch Experiment Runner'
    )
    
    parser.add_argument('--output-dir', type=str, default='./outputs/batch',
                       help='Output directory for results')
    parser.add_argument('--datasets', nargs='+', default=None,
                       choices=['mnist', 'fmnist', 'cifar10', 'svhn'],
                       help='Datasets to run (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with reduced rounds')
    
    args = parser.parse_args()
    
    # Modify for quick test
    if args.quick:
        PAPER_EXPERIMENTS['num_rounds'] = 10
        print("Running in QUICK mode (10 rounds)")
    
    run_batch_experiments(
        output_dir=args.output_dir,
        datasets=args.datasets,
        seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main()
