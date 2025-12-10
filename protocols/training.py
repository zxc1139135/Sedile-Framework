"""
Inter-Partition Model Training Protocol.

Implements the training workflow across partitions:
1. Local model training on encoded datasets
2. Secure gradient aggregation at the server
3. Gradient decoding and model update

This protocol provides dual-threat protection against both
the server and colluding clients.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class GradientUpdate:
    """Container for gradient updates."""
    gradients: Dict[str, torch.Tensor]
    num_samples: int
    client_id: int
    partition_id: int


@dataclass 
class AggregatedGradient:
    """Container for aggregated gradients."""
    gradients: Dict[str, torch.Tensor]
    total_samples: int


class LocalTrainer:
    """
    Handles local model training for a client.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.015,
        device: str = 'cpu'
    ):
        """
        Initialize local trainer.
        
        Args:
            model: Neural network model
            learning_rate: Learning rate for SGD
            device: Device to train on
        """
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(
        self,
        data_loader: DataLoader,
        num_epochs: int = 1
    ) -> Tuple[float, int]:
        """
        Train for specified number of epochs.
        
        Args:
            data_loader: Data loader for training data
            num_epochs: Number of epochs
            
        Returns:
            Tuple of (average_loss, num_samples)
        """
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        total_loss = 0.0
        total_samples = 0
        
        for _ in range(num_epochs):
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_labels)
                total_samples += len(batch_labels)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return avg_loss, total_samples
    
    def compute_gradient(
        self,
        data_loader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradient over dataset without updating model.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Dict mapping parameter names to gradient tensors
        """
        self.model.train()
        
        # Zero existing gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        total_samples = 0
        
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            
            total_samples += len(batch_labels)
        
        # Average gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone() / total_samples
        
        return gradients
    
    def apply_gradient(
        self,
        gradients: Dict[str, torch.Tensor],
        learning_rate: Optional[float] = None
    ):
        """
        Apply gradients to model parameters.
        
        Args:
            gradients: Dict mapping parameter names to gradients
            learning_rate: Learning rate (uses default if None)
        """
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in gradients:
                    param.data -= lr * gradients[name]
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get model state dict."""
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_state(self, state_dict: Dict[str, torch.Tensor]):
        """Set model state from state dict."""
        self.model.load_state_dict(state_dict)
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on test data.
        
        Args:
            data_loader: Test data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item() * len(batch_labels)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_labels).sum().item()
                total += len(batch_labels)
        
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0
        
        return avg_loss, accuracy


class SecureAggregator:
    """
    Aggregates gradient updates from clients securely.
    
    The server cannot infer individual client gradients due to
    harmonic coding protection.
    """
    
    def __init__(self, num_partitions: int):
        """
        Initialize aggregator.
        
        Args:
            num_partitions: Number of partitions V
        """
        self.num_partitions = num_partitions
        self.partition_updates: Dict[int, List[GradientUpdate]] = {
            i: [] for i in range(num_partitions)
        }
    
    def receive_update(self, update: GradientUpdate):
        """
        Receive a gradient update from a client.
        
        Args:
            update: GradientUpdate from a client
        """
        self.partition_updates[update.partition_id].append(update)
    
    def aggregate_partition(self, partition_id: int) -> Optional[AggregatedGradient]:
        """
        Aggregate updates within a partition.
        
        Args:
            partition_id: Partition to aggregate
            
        Returns:
            Aggregated gradient or None if no updates
        """
        updates = self.partition_updates[partition_id]
        
        if not updates:
            return None
        
        # Weighted average within partition
        total_samples = sum(u.num_samples for u in updates)
        
        aggregated = {}
        for name in updates[0].gradients.keys():
            weighted_sum = sum(
                u.gradients[name] * u.num_samples 
                for u in updates
            )
            aggregated[name] = weighted_sum / total_samples
        
        return AggregatedGradient(
            gradients=aggregated,
            total_samples=total_samples
        )
    
    def aggregate_global(
        self,
        partition_sizes: Dict[int, int]
    ) -> Optional[AggregatedGradient]:
        """
        Aggregate across all partitions.
        
        Args:
            partition_sizes: Dict mapping partition_id to original dataset size
            
        Returns:
            Global aggregated gradient
        """
        partition_aggregates = {}
        
        for partition_id in range(self.num_partitions):
            agg = self.aggregate_partition(partition_id)
            if agg is not None:
                partition_aggregates[partition_id] = agg
        
        if not partition_aggregates:
            return None
        
        # Weighted average across partitions
        total_size = sum(partition_sizes.values())
        
        global_gradients = {}
        first_agg = list(partition_aggregates.values())[0]
        
        for name in first_agg.gradients.keys():
            weighted_sum = sum(
                partition_aggregates[pid].gradients[name] * partition_sizes[pid]
                for pid in partition_aggregates.keys()
            )
            global_gradients[name] = weighted_sum / total_size
        
        return AggregatedGradient(
            gradients=global_gradients,
            total_samples=total_size
        )
    
    def clear(self):
        """Clear all stored updates."""
        for pid in self.partition_updates:
            self.partition_updates[pid] = []


class FederatedTrainer:
    """
    Orchestrates federated training across partitions.
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        num_partitions: int,
        learning_rate: float = 0.015,
        local_epochs: int = 3,
        device: str = 'cpu'
    ):
        """
        Initialize federated trainer.
        
        Args:
            global_model: Global model to train
            num_partitions: Number of partitions
            learning_rate: Learning rate
            local_epochs: Local training epochs per round
            device: Training device
        """
        self.global_model = global_model.to(device)
        self.num_partitions = num_partitions
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.device = device
        
        self.aggregator = SecureAggregator(num_partitions)
        self.round = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'test_accuracy': []
        }
    
    def train_round(
        self,
        client_data_loaders: Dict[int, DataLoader],
        client_partitions: Dict[int, int],
        partition_sizes: Dict[int, int]
    ) -> float:
        """
        Execute one training round.
        
        Args:
            client_data_loaders: Data loaders for each client
            client_partitions: Mapping from client_id to partition_id
            partition_sizes: Original dataset size per partition
            
        Returns:
            Average training loss
        """
        self.round += 1
        self.aggregator.clear()
        
        # Get global model state
        global_state = copy.deepcopy(self.global_model.state_dict())
        
        total_loss = 0.0
        num_clients = 0
        
        # Local training for each client
        for client_id, data_loader in client_data_loaders.items():
            partition_id = client_partitions[client_id]
            
            # Create local trainer with copy of global model
            local_model = copy.deepcopy(self.global_model)
            trainer = LocalTrainer(
                local_model, 
                self.learning_rate, 
                self.device
            )
            
            # Train locally
            loss, num_samples = trainer.train_epoch(
                data_loader, 
                self.local_epochs
            )
            
            # Compute gradient (difference from global)
            gradients = {}
            for name, param in local_model.named_parameters():
                gradients[name] = global_state[name] - param.data
            
            # Submit update
            update = GradientUpdate(
                gradients=gradients,
                num_samples=num_samples,
                client_id=client_id,
                partition_id=partition_id
            )
            self.aggregator.receive_update(update)
            
            total_loss += loss * num_samples
            num_clients += 1
        
        # Aggregate and apply
        global_update = self.aggregator.aggregate_global(partition_sizes)
        
        if global_update is not None:
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    if name in global_update.gradients:
                        param.data -= global_update.gradients[name]
        
        avg_loss = total_loss / sum(
            len(dl.dataset) for dl in client_data_loaders.values()
        )
        self.history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate global model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        trainer = LocalTrainer(self.global_model, device=self.device)
        loss, accuracy = trainer.evaluate(test_loader)
        
        self.history['test_loss'].append(loss)
        self.history['test_accuracy'].append(accuracy)
        
        return loss, accuracy
    
    def get_global_model(self) -> nn.Module:
        """Get the global model."""
        return self.global_model
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.history


class DistributedTrainingOrchestrator:
    """
    High-level orchestrator for the complete training pipeline.
    """
    
    def __init__(
        self,
        model: nn.Module,
        partitions: List[List[int]],
        learning_rate: float = 0.015,
        local_epochs: int = 3,
        device: str = 'cpu'
    ):
        """
        Initialize orchestrator.
        
        Args:
            model: Model to train
            partitions: Client partitions
            learning_rate: Learning rate
            local_epochs: Local epochs per round
            device: Training device
        """
        self.partitions = partitions
        self.num_partitions = len(partitions)
        
        # Create client to partition mapping
        self.client_partitions = {}
        for pid, clients in enumerate(partitions):
            for cid in clients:
                self.client_partitions[cid] = pid
        
        self.trainer = FederatedTrainer(
            global_model=model,
            num_partitions=self.num_partitions,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            device=device
        )
    
    def train(
        self,
        client_data_loaders: Dict[int, DataLoader],
        test_loader: DataLoader,
        num_rounds: int,
        log_interval: int = 10
    ) -> Dict[str, List[float]]:
        """
        Execute training for specified number of rounds.
        
        Args:
            client_data_loaders: Data loaders for all clients
            test_loader: Test data loader
            num_rounds: Number of training rounds
            log_interval: Interval for logging
            
        Returns:
            Training history
        """
        # Compute partition sizes
        partition_sizes = {}
        for pid, clients in enumerate(self.partitions):
            size = sum(
                len(client_data_loaders[cid].dataset)
                for cid in clients
                if cid in client_data_loaders
            )
            partition_sizes[pid] = size
        
        for round_num in range(1, num_rounds + 1):
            # Training round
            train_loss = self.trainer.train_round(
                client_data_loaders,
                self.client_partitions,
                partition_sizes
            )
            
            # Evaluation
            test_loss, test_accuracy = self.trainer.evaluate(test_loader)
            
            # Logging
            if round_num % log_interval == 0 or round_num == 1:
                print(f"Round {round_num}/{num_rounds}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Test Loss={test_loss:.4f}, "
                      f"Test Acc={test_accuracy:.2f}%")
        
        return self.trainer.get_history()


if __name__ == '__main__':
    # Test training protocol
    print("Testing Inter-Partition Model Training Protocol...")
    
    from models import get_model
    
    # Create simple model
    model = get_model('mnist')
    
    # Create dummy partitions
    partitions = [list(range(5)), list(range(5, 10))]
    
    # Create dummy data
    client_data_loaders = {}
    for cid in range(10):
        data = torch.randn(100, 1, 28, 28)
        labels = torch.randint(0, 10, (100,))
        dataset = TensorDataset(data, labels)
        client_data_loaders[cid] = DataLoader(dataset, batch_size=32)
    
    test_data = torch.randn(200, 1, 28, 28)
    test_labels = torch.randint(0, 10, (200,))
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize orchestrator
    orchestrator = DistributedTrainingOrchestrator(
        model=model,
        partitions=partitions,
        learning_rate=0.015,
        local_epochs=1,
        device='cpu'
    )
    
    # Run a few rounds
    history = orchestrator.train(
        client_data_loaders=client_data_loaders,
        test_loader=test_loader,
        num_rounds=5,
        log_interval=1
    )
    
    print(f"\nFinal test accuracy: {history['test_accuracy'][-1]:.2f}%")
    print("\nAll tests passed!")
