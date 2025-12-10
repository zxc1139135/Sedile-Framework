"""
Data Loading and Non-IID Distribution.

Handles loading of datasets and simulating non-IID data distributions:
- Dirichlet distribution for label skew
- Pathological distribution for extreme non-IID scenarios

Supported datasets: MNIST, Fashion-MNIST, CIFAR-10, SVHN
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Optional
import os


def get_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and test transforms for each dataset.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'fmnist', 'cifar10', 'svhn')
        
    Returns:
        Tuple of (train_transform, test_transform)
    """
    if dataset_name in ['mnist', 'fmnist']:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_transform = train_transform
        
    elif dataset_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
    elif dataset_name == 'svhn':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        test_transform = train_transform
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_transform, test_transform


def load_dataset(
    dataset_name: str, 
    data_dir: str = './data/raw'
) -> Tuple[Dataset, Dataset]:
    """
    Load a dataset.
    
    Args:
        dataset_name: Name of dataset
        data_dir: Directory to store/load data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_transform, test_transform = get_transforms(dataset_name)
    
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=test_transform
        )
        
    elif dataset_name == 'fmnist':
        train_dataset = datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.FashionMNIST(
            data_dir, train=False, download=True, transform=test_transform
        )
        
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=test_transform
        )
        
    elif dataset_name == 'svhn':
        train_dataset = datasets.SVHN(
            data_dir, split='train', download=True, transform=train_transform
        )
        test_dataset = datasets.SVHN(
            data_dir, split='test', download=True, transform=test_transform
        )
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset


def get_targets(dataset: Dataset) -> np.ndarray:
    """Extract targets/labels from dataset."""
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.numpy()
        return np.array(targets)
    elif hasattr(dataset, 'labels'):
        return np.array(dataset.labels)
    else:
        # Iterate through dataset
        return np.array([dataset[i][1] for i in range(len(dataset))])


class DirichletPartitioner:
    """
    Partition data using Dirichlet distribution.
    
    Simulates label distribution skew where each client has
    an imbalanced distribution of labels controlled by alpha.
    """
    
    def __init__(self, alpha: float, num_clients: int, num_classes: int = 10):
        """
        Initialize partitioner.
        
        Args:
            alpha: Dirichlet concentration parameter (smaller = more skewed)
            num_clients: Number of clients to partition data among
            num_classes: Number of label classes
        """
        self.alpha = alpha
        self.num_clients = num_clients
        self.num_classes = num_classes
    
    def partition(
        self, 
        dataset: Dataset, 
        seed: int = 42
    ) -> List[List[int]]:
        """
        Partition dataset indices among clients.
        
        Args:
            dataset: Dataset to partition
            seed: Random seed for reproducibility
            
        Returns:
            List of index lists, one per client
        """
        np.random.seed(seed)
        
        targets = get_targets(dataset)
        n_samples = len(targets)
        
        # Group indices by class
        class_indices = {c: np.where(targets == c)[0] for c in range(self.num_classes)}
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(
            [self.alpha] * self.num_clients, 
            self.num_classes
        )
        
        # Assign samples to clients
        client_indices = [[] for _ in range(self.num_clients)]
        
        for c in range(self.num_classes):
            indices = class_indices[c]
            np.random.shuffle(indices)
            
            # Split indices according to proportions
            splits = (proportions[c] * len(indices)).astype(int)
            splits[-1] = len(indices) - splits[:-1].sum()  # Handle rounding
            
            start = 0
            for client_id, size in enumerate(splits):
                if size > 0:
                    client_indices[client_id].extend(indices[start:start + size].tolist())
                    start += size
        
        return client_indices
    
    def get_distribution(
        self, 
        client_indices: List[List[int]], 
        dataset: Dataset
    ) -> np.ndarray:
        """
        Get label distribution for each client.
        
        Args:
            client_indices: List of index lists per client
            dataset: Original dataset
            
        Returns:
            Array of shape (num_clients, num_classes) with proportions
        """
        targets = get_targets(dataset)
        distributions = np.zeros((self.num_clients, self.num_classes))
        
        for client_id, indices in enumerate(client_indices):
            if len(indices) > 0:
                client_targets = targets[indices]
                for c in range(self.num_classes):
                    distributions[client_id, c] = (client_targets == c).sum()
                distributions[client_id] /= distributions[client_id].sum()
        
        return distributions


class PathologicalPartitioner:
    """
    Partition data using pathological distribution.
    
    Each client only has data from kappa classes,
    simulating extreme non-IID scenarios.
    """
    
    def __init__(self, kappa: int, num_clients: int, num_classes: int = 10):
        """
        Initialize partitioner.
        
        Args:
            kappa: Number of classes per client
            num_clients: Number of clients
            num_classes: Total number of classes
        """
        if kappa > num_classes:
            raise ValueError(f"kappa ({kappa}) cannot exceed num_classes ({num_classes})")
        
        self.kappa = kappa
        self.num_clients = num_clients
        self.num_classes = num_classes
    
    def partition(
        self, 
        dataset: Dataset, 
        seed: int = 42
    ) -> List[List[int]]:
        """
        Partition dataset indices among clients.
        
        Each client receives samples from exactly kappa classes.
        """
        np.random.seed(seed)
        
        targets = get_targets(dataset)
        
        # Group indices by class
        class_indices = {c: np.where(targets == c)[0].tolist() 
                        for c in range(self.num_classes)}
        
        # Shuffle indices within each class
        for c in class_indices:
            np.random.shuffle(class_indices[c])
        
        # Assign classes to clients
        client_indices = [[] for _ in range(self.num_clients)]
        
        # Create shards: divide each class into (num_clients * kappa / num_classes) shards
        shards_per_class = max(1, (self.num_clients * self.kappa) // self.num_classes)
        
        all_shards = []
        for c in range(self.num_classes):
            indices = class_indices[c]
            shard_size = len(indices) // shards_per_class
            for i in range(shards_per_class):
                start = i * shard_size
                end = start + shard_size if i < shards_per_class - 1 else len(indices)
                all_shards.append((c, indices[start:end]))
        
        # Randomly assign shards to clients
        np.random.shuffle(all_shards)
        
        shard_idx = 0
        for client_id in range(self.num_clients):
            assigned_classes = set()
            while len(assigned_classes) < self.kappa and shard_idx < len(all_shards):
                class_label, indices = all_shards[shard_idx]
                if class_label not in assigned_classes or len(assigned_classes) == self.kappa:
                    client_indices[client_id].extend(indices)
                    assigned_classes.add(class_label)
                shard_idx += 1
        
        return client_indices
    
    def get_distribution(
        self, 
        client_indices: List[List[int]], 
        dataset: Dataset
    ) -> np.ndarray:
        """Get label distribution for each client."""
        targets = get_targets(dataset)
        distributions = np.zeros((self.num_clients, self.num_classes))
        
        for client_id, indices in enumerate(client_indices):
            if len(indices) > 0:
                client_targets = targets[indices]
                for c in range(self.num_classes):
                    distributions[client_id, c] = (client_targets == c).sum()
                total = distributions[client_id].sum()
                if total > 0:
                    distributions[client_id] /= total
        
        return distributions


class ClientDataset(Dataset):
    """Dataset wrapper for a single client's data."""
    
    def __init__(self, dataset: Dataset, indices: List[int]):
        """
        Initialize client dataset.
        
        Args:
            dataset: Original full dataset
            indices: Indices belonging to this client
        """
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all data and labels as tensors."""
        data_list = []
        label_list = []
        
        for idx in self.indices:
            data, label = self.dataset[idx]
            data_list.append(data)
            label_list.append(label)
        
        return torch.stack(data_list), torch.tensor(label_list)


class FederatedDataLoader:
    """
    Manages data loading for federated learning setup.
    """
    
    def __init__(
        self,
        dataset_name: str,
        num_clients: int,
        distribution_type: str = 'dirichlet',
        distribution_param: float = 0.1,
        batch_size: int = 32,
        data_dir: str = './data/raw',
        seed: int = 42
    ):
        """
        Initialize federated data loader.
        
        Args:
            dataset_name: Name of dataset
            num_clients: Number of clients
            distribution_type: 'dirichlet' or 'pathological'
            distribution_param: alpha for dirichlet, kappa for pathological
            batch_size: Batch size for training
            data_dir: Data directory
            seed: Random seed
        """
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.seed = seed
        
        # Load dataset
        self.train_dataset, self.test_dataset = load_dataset(dataset_name, data_dir)
        
        # Create partitioner
        if distribution_type == 'dirichlet':
            self.partitioner = DirichletPartitioner(
                alpha=distribution_param,
                num_clients=num_clients
            )
        else:
            self.partitioner = PathologicalPartitioner(
                kappa=int(distribution_param),
                num_clients=num_clients
            )
        
        # Partition data
        self.client_indices = self.partitioner.partition(self.train_dataset, seed)
        self.distributions = self.partitioner.get_distribution(
            self.client_indices, self.train_dataset
        )
        
        # Create client datasets
        self.client_datasets = [
            ClientDataset(self.train_dataset, indices)
            for indices in self.client_indices
        ]
    
    def get_client_loader(self, client_id: int) -> DataLoader:
        """Get data loader for a specific client."""
        return DataLoader(
            self.client_datasets[client_id],
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 4,
            shuffle=False
        )
    
    def get_client_distribution(self, client_id: int) -> np.ndarray:
        """Get label distribution for a client."""
        return self.distributions[client_id]
    
    def get_client_data_size(self, client_id: int) -> int:
        """Get number of samples for a client."""
        return len(self.client_indices[client_id])
    
    def get_all_distributions(self) -> np.ndarray:
        """Get distributions for all clients."""
        return self.distributions


def create_iid_partition(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42
) -> List[List[int]]:
    """
    Create IID partition of dataset.
    
    Args:
        dataset: Dataset to partition
        num_clients: Number of clients
        seed: Random seed
        
    Returns:
        List of index lists, one per client
    """
    np.random.seed(seed)
    
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    return np.array_split(indices, num_clients)


if __name__ == '__main__':
    # Test data loading and partitioning
    print("Testing data loading...")
    
    # Test Dirichlet partitioning
    loader = FederatedDataLoader(
        dataset_name='mnist',
        num_clients=10,
        distribution_type='dirichlet',
        distribution_param=0.1,
        batch_size=32
    )
    
    print(f"Number of clients: {loader.num_clients}")
    print(f"Client 0 data size: {loader.get_client_data_size(0)}")
    print(f"Client 0 distribution: {loader.get_client_distribution(0)}")
    
    # Test client loader
    client_loader = loader.get_client_loader(0)
    batch = next(iter(client_loader))
    print(f"Batch shape: {batch[0].shape}, {batch[1].shape}")
    
    # Test pathological partitioning
    loader_path = FederatedDataLoader(
        dataset_name='mnist',
        num_clients=10,
        distribution_type='pathological',
        distribution_param=2,
        batch_size=32
    )
    
    print(f"\nPathological (kappa=2):")
    print(f"Client 0 distribution: {loader_path.get_client_distribution(0)}")
    
    print("\nAll tests passed!")
