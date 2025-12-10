"""
LeNet-5 Model Implementation.

Classic convolutional neural network architecture.
Used for Fashion-MNIST dataset as specified in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5 Architecture.
    
    Architecture:
        Conv(6) -> ReLU -> MaxPool -> Conv(16) -> ReLU -> MaxPool ->
        FC(120) -> ReLU -> FC(84) -> ReLU -> FC(10)
    """
    
    def __init__(
        self, 
        input_channels: int = 1, 
        num_classes: int = 10
    ):
        """
        Initialize LeNet-5.
        
        Args:
            input_channels: Number of input channels (1 for grayscale)
            num_classes: Number of output classes
        """
        super(LeNet5, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class LeNet5BN(nn.Module):
    """
    LeNet-5 with Batch Normalization.
    
    Enhanced version with batch normalization for better training stability.
    """
    
    def __init__(
        self, 
        input_channels: int = 1, 
        num_classes: int = 10
    ):
        super(LeNet5BN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


def create_lenet(
    input_channels: int = 1,
    num_classes: int = 10,
    use_bn: bool = False
) -> nn.Module:
    """
    Factory function to create LeNet model.
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        use_bn: Whether to use batch normalization
        
    Returns:
        LeNet model instance
    """
    if use_bn:
        return LeNet5BN(input_channels, num_classes)
    return LeNet5(input_channels, num_classes)


if __name__ == '__main__':
    # Test LeNet
    model = create_lenet()
    print(f"Model architecture:\n{model}")
    
    # Test forward pass
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
