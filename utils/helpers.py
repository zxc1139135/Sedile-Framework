"""
Utility functions for Sedile framework.

Provides helper functions for:
- Random seed setting
- Logging configuration
- Metrics computation
- Result saving and loading
"""

import os
import json
import random
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(
    log_dir: str = './logs',
    log_level: int = logging.INFO,
    experiment_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        experiment_name: Name for the experiment
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = experiment_name if experiment_name else 'sedile'
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    logger = logging.getLogger('sedile')
    logger.setLevel(log_level)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """Compute classification accuracy."""
    if predictions.dim() > 1:
        _, predicted = predictions.max(1)
    else:
        predicted = predictions
    
    correct = predicted.eq(labels).sum().item()
    total = labels.size(0)
    
    return 100.0 * correct / total


def compute_f1_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10
) -> Dict[str, float]:
    """Compute F1 score (macro and micro)."""
    if predictions.dim() > 1:
        _, predicted = predictions.max(1)
    else:
        predicted = predictions
    
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    
    for c in range(num_classes):
        tp = np.sum((predicted == c) & (labels == c))
        fp = np.sum((predicted == c) & (labels != c))
        fn = np.sum((predicted != c) & (labels == c))
        
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-10)
    macro_f1 = np.mean(f1_per_class)
    micro_f1 = np.sum(predicted == labels) / len(labels)
    
    return {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'per_class_f1': f1_per_class.tolist()
    }


class MetricsTracker:
    """Tracks and stores training metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def add_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Add a scalar metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def add_dict(self, metrics_dict: Dict[str, float], step: int):
        """Add multiple metrics at once."""
        for name, value in metrics_dict.items():
            self.add_scalar(name, value, step)
    
    def get_metric(self, name: str) -> List[float]:
        """Get all values for a metric."""
        return self.metrics.get(name, [])
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for a metric."""
        values = self.metrics.get(name, [])
        return values[-1] if values else None
    
    def get_best(self, name: str, mode: str = 'max') -> Optional[float]:
        """Get best value for a metric."""
        values = self.metrics.get(name, [])
        if not values:
            return None
        return max(values) if mode == 'max' else min(values)
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary."""
        return self.metrics.copy()
    
    def save(self, path: str):
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'MetricsTracker':
        """Load metrics from JSON file."""
        tracker = cls()
        with open(path, 'r') as f:
            tracker.metrics = json.load(f)
        return tracker


class ResultsSaver:
    """Handles saving and loading experimental results."""
    
    def __init__(self, output_dir: str = './outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_config(self, config: Dict[str, Any], name: str = 'config'):
        """Save configuration dictionary."""
        path = os.path.join(self.output_dir, f'{name}.json')
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def save_metrics(self, metrics: Dict[str, List[float]], name: str = 'metrics'):
        """Save training metrics."""
        path = os.path.join(self.output_dir, f'{name}.json')
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_model(self, model: torch.nn.Module, name: str = 'model'):
        """Save model state dict."""
        path = os.path.join(self.output_dir, f'{name}.pt')
        torch.save(model.state_dict(), path)
    
    def load_model(self, model: torch.nn.Module, name: str = 'model'):
        """Load model state dict."""
        path = os.path.join(self.output_dir, f'{name}.pt')
        model.load_state_dict(torch.load(path))
        return model
    
    def save_numpy(self, array: np.ndarray, name: str):
        """Save numpy array."""
        path = os.path.join(self.output_dir, f'{name}.npy')
        np.save(path, array)
    
    def load_numpy(self, name: str) -> np.ndarray:
        """Load numpy array."""
        path = os.path.join(self.output_dir, f'{name}.npy')
        return np.load(path)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = 'Training Progress'
):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
    if 'test_loss' in history:
        axes[0].plot(history['test_loss'], label='Test Loss')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if 'test_accuracy' in history:
        axes[1].plot(history['test_accuracy'], label='Test Accuracy', color='green')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_distribution_heatmap(
    distributions: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'Client Data Distribution'
):
    """Plot client data distribution as heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(distributions, aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Class')
    ax.set_ylabel('Client')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_device() -> str:
    """Get available device (cuda or cpu)."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f'{hours}h {minutes}m {secs}s'
    elif minutes > 0:
        return f'{minutes}m {secs}s'
    else:
        return f'{secs}s'
