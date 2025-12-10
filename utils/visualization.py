"""
Visualization Utilities.

Provides functions for plotting:
- Training curves
- Data distribution analysis
- Client partition visualization
- Performance comparisons
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import seaborn as sns


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = 'Training Progress'
):
    """
    Plot training loss and accuracy curves.
    
    Args:
        history: Dict with 'train_loss', 'test_loss', 'test_accuracy' lists
        save_path: Path to save figure (displays if None)
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    rounds = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss subplot
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(rounds, history['train_loss'], label='Train Loss', linewidth=2)
    if 'test_loss' in history:
        ax1.plot(rounds, history['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy subplot
    ax2 = axes[1]
    if 'test_accuracy' in history:
        ax2.plot(rounds, history['test_accuracy'], label='Test Accuracy', 
                linewidth=2, color='green')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_client_distributions(
    distributions: np.ndarray,
    client_ids: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    title: str = 'Client Data Distributions'
):
    """
    Plot data distribution for each client as a heatmap.
    
    Args:
        distributions: Array of shape (num_clients, num_classes)
        client_ids: List of client IDs (uses indices if None)
        save_path: Path to save figure
        title: Plot title
    """
    num_clients, num_classes = distributions.shape
    
    if client_ids is None:
        client_ids = list(range(num_clients))
    
    fig, ax = plt.subplots(figsize=(10, max(6, num_clients * 0.3)))
    
    im = ax.imshow(distributions, aspect='auto', cmap='YlOrRd')
    
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax.set_yticks(range(num_clients))
    ax.set_yticklabels([f'Client {i}' for i in client_ids])
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Client')
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Proportion')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_partition_distributions(
    distributions: np.ndarray,
    partition_assignments: List[List[int]],
    save_path: Optional[str] = None,
    title: str = 'Partition Data Distributions'
):
    """
    Plot aggregated distribution for each partition.
    
    Args:
        distributions: Array of shape (num_clients, num_classes)
        partition_assignments: List of client ID lists per partition
        save_path: Path to save figure
        title: Plot title
    """
    num_partitions = len(partition_assignments)
    num_classes = distributions.shape[1]
    
    # Compute partition distributions
    partition_dists = np.zeros((num_partitions, num_classes))
    for pid, clients in enumerate(partition_assignments):
        if len(clients) > 0:
            partition_dists[pid] = np.mean(distributions[clients], axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    x = np.arange(num_classes)
    width = 0.8 / num_partitions
    
    colors = plt.cm.Set2(np.linspace(0, 1, num_partitions))
    
    for pid in range(num_partitions):
        offset = (pid - num_partitions / 2 + 0.5) * width
        ax.bar(x + offset, partition_dists[pid], width, 
               label=f'Partition {pid}', color=colors[pid])
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Proportion')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(num_classes)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_convergence_comparison(
    results: Dict[str, Dict[str, List[float]]],
    metric: str = 'test_accuracy',
    save_path: Optional[str] = None,
    title: str = 'Convergence Comparison'
):
    """
    Compare convergence across different configurations.
    
    Args:
        results: Dict mapping config names to their history dicts
        metric: Metric to compare
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for idx, (name, history) in enumerate(results.items()):
        if metric in history:
            rounds = range(1, len(history[metric]) + 1)
            ax.plot(rounds, history[metric], label=name, 
                   linewidth=2, color=colors[idx])
    
    ax.set_xlabel('Round')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_data_sizes(
    client_sizes: List[int],
    partition_assignments: Optional[List[List[int]]] = None,
    save_path: Optional[str] = None,
    title: str = 'Client Data Sizes'
):
    """
    Plot data size distribution across clients.
    
    Args:
        client_sizes: List of data sizes per client
        partition_assignments: Optional partition groupings
        save_path: Path to save figure
        title: Plot title
    """
    num_clients = len(client_sizes)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    if partition_assignments:
        num_partitions = len(partition_assignments)
        colors = plt.cm.Set2(np.linspace(0, 1, num_partitions))
        
        client_colors = ['gray'] * num_clients
        for pid, clients in enumerate(partition_assignments):
            for cid in clients:
                if cid < num_clients:
                    client_colors[cid] = colors[pid]
        
        bars = ax.bar(range(num_clients), client_sizes, color=client_colors)
        
        # Create legend
        patches = [mpatches.Patch(color=colors[i], label=f'Partition {i}') 
                  for i in range(num_partitions)]
        ax.legend(handles=patches)
    else:
        ax.bar(range(num_clients), client_sizes)
    
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_similarity_matrix(
    distributions: np.ndarray,
    metric: str = 'euclidean',
    save_path: Optional[str] = None,
    title: str = 'Client Similarity Matrix'
):
    """
    Plot pairwise similarity matrix between clients.
    
    Args:
        distributions: Array of shape (num_clients, num_classes)
        metric: Similarity metric ('euclidean', 'cosine')
        save_path: Path to save figure
        title: Plot title
    """
    num_clients = len(distributions)
    
    # Compute pairwise similarities
    similarity_matrix = np.zeros((num_clients, num_clients))
    
    for i in range(num_clients):
        for j in range(num_clients):
            if metric == 'euclidean':
                similarity_matrix[i, j] = np.sqrt(np.sum((distributions[i] - distributions[j]) ** 2))
            elif metric == 'cosine':
                dot = np.dot(distributions[i], distributions[j])
                norm_i = np.linalg.norm(distributions[i])
                norm_j = np.linalg.norm(distributions[j])
                if norm_i > 0 and norm_j > 0:
                    similarity_matrix[i, j] = dot / (norm_i * norm_j)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cmap = 'RdYlBu' if metric == 'cosine' else 'RdYlBu_r'
    im = ax.imshow(similarity_matrix, cmap=cmap)
    
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Client ID')
    ax.set_title(f'{title} ({metric})')
    
    plt.colorbar(im, ax=ax, label='Distance' if metric == 'euclidean' else 'Similarity')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_accuracy_vs_heterogeneity(
    results: List[Tuple[float, float]],
    labels: List[str],
    save_path: Optional[str] = None,
    title: str = 'Accuracy vs Data Heterogeneity'
):
    """
    Plot accuracy against data heterogeneity level.
    
    Args:
        results: List of (heterogeneity_param, accuracy) tuples
        labels: Labels for each point
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = [r[0] for r in results]
    y = [r[1] for r in results]
    
    ax.scatter(x, y, s=100, zorder=3)
    ax.plot(x, y, '--', alpha=0.5, zorder=2)
    
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]), textcoords="offset points", 
                   xytext=(0, 10), ha='center')
    
    ax.set_xlabel('Heterogeneity Parameter')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_experiment_report(
    experiment_name: str,
    config: Dict,
    history: Dict[str, List[float]],
    distributions: np.ndarray,
    partitions: List[List[int]],
    output_dir: str
):
    """
    Create comprehensive experiment report with all plots.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        history: Training history
        distributions: Client data distributions
        partitions: Client partition assignments
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Training curves
    plot_training_curves(
        history,
        save_path=os.path.join(output_dir, 'training_curves.png'),
        title=f'{experiment_name} - Training Progress'
    )
    
    # Client distributions
    plot_client_distributions(
        distributions,
        save_path=os.path.join(output_dir, 'client_distributions.png'),
        title=f'{experiment_name} - Client Data Distributions'
    )
    
    # Partition distributions
    plot_partition_distributions(
        distributions,
        partitions,
        save_path=os.path.join(output_dir, 'partition_distributions.png'),
        title=f'{experiment_name} - Partition Data Distributions'
    )
    
    # Similarity matrix
    plot_similarity_matrix(
        distributions,
        save_path=os.path.join(output_dir, 'similarity_matrix.png'),
        title=f'{experiment_name} - Client Similarity'
    )
    
    print(f"Report saved to {output_dir}")


if __name__ == '__main__':
    # Test visualization
    print("Testing visualization utilities...")
    
    # Generate test data
    num_clients = 20
    num_classes = 10
    
    # Random distributions
    distributions = np.random.dirichlet([0.5] * num_classes, size=num_clients)
    
    # Random partitions
    partitions = [list(range(10)), list(range(10, 20))]
    
    # Random history
    history = {
        'train_loss': list(np.linspace(2.0, 0.1, 100)),
        'test_loss': list(np.linspace(2.2, 0.15, 100)),
        'test_accuracy': list(np.linspace(10, 95, 100))
    }
    
    print("Generating test plots...")
    
    # Test each plot function (would display if not in test mode)
    # plot_training_curves(history, title='Test Training Curves')
    # plot_client_distributions(distributions, title='Test Client Distributions')
    
    print("Visualization tests completed!")
