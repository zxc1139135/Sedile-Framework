"""
Multi-Layer Perceptron (MLP) Model.

Architecture: Two hidden layers with 256 nodes each.
Used for MNIST dataset as specified in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with ReLU activations.
    
    Architecture:
        Input -> FC(256) -> ReLU -> FC(256) -> ReLU -> FC(10) -> Output
    """
    
    def __init__(
        self, 
        input_dim: int = 784, 
        hidden_dims: List[int] = [256, 256],
        num_classes: int = 10
    ):
        """
        Initialize MLP.
        
        Args:
            input_dim: Dimension of flattened input (784 for MNIST)
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.network(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the penultimate layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        x = x.view(x.size(0), -1)
        
        # Pass through all but last layer
        for layer in list(self.network.children())[:-1]:
            x = layer(x)
        
        return x


def create_mlp(
    input_channels: int = 1,
    input_size: int = 28,
    num_classes: int = 10,
    hidden_dims: List[int] = [256, 256]
) -> MLP:
    """
    Factory function to create MLP model.
    
    Args:
        input_channels: Number of input channels (1 for grayscale)
        input_size: Height/width of input image
        num_classes: Number of output classes
        hidden_dims: Hidden layer dimensions
        
    Returns:
        MLP model instance
    """
    input_dim = input_channels * input_size * input_size
    return MLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)


if __name__ == '__main__':
    # Test MLP
    model = create_mlp()
    print(f"Model architecture:\n{model}")
    
    # Test forward pass
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
