"""
VGG Model Implementation.

VGG-style architecture adapted for CIFAR-10.
Used for CIFAR-10 dataset as specified in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# VGG configurations
VGG_CONFIGS = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """
    VGG Architecture for CIFAR-10.
    
    Adapted from original VGG for 32x32 images.
    """
    
    def __init__(
        self, 
        config: List,
        input_channels: int = 3,
        num_classes: int = 10,
        use_bn: bool = True
    ):
        """
        Initialize VGG.
        
        Args:
            config: List defining layer configuration
            input_channels: Number of input channels
            num_classes: Number of output classes
            use_bn: Whether to use batch normalization
        """
        super(VGG, self).__init__()
        
        self.features = self._make_layers(config, input_channels, use_bn)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _make_layers(
        self, 
        config: List, 
        input_channels: int,
        use_bn: bool
    ) -> nn.Sequential:
        """Build convolutional layers."""
        layers = []
        in_channels = input_channels
        
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if use_bn:
                    layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv, nn.ReLU(inplace=True)])
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class VGGSmall(nn.Module):
    """
    Simplified VGG for CIFAR-10.
    
    Smaller architecture suitable for 32x32 images.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10
    ):
        super(VGGSmall, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def create_vgg(
    arch: str = 'vgg11',
    input_channels: int = 3,
    num_classes: int = 10,
    use_bn: bool = True
) -> nn.Module:
    """
    Factory function to create VGG model.
    
    Args:
        arch: VGG architecture variant ('vgg11', 'vgg13', 'vgg16', 'vgg19', 'small')
        input_channels: Number of input channels
        num_classes: Number of output classes
        use_bn: Whether to use batch normalization
        
    Returns:
        VGG model instance
    """
    if arch == 'small':
        return VGGSmall(input_channels, num_classes)
    
    if arch not in VGG_CONFIGS:
        raise ValueError(f"Unknown VGG architecture: {arch}")
    
    return VGG(VGG_CONFIGS[arch], input_channels, num_classes, use_bn)


if __name__ == '__main__':
    # Test VGG
    model = create_vgg('vgg11')
    print(f"Model architecture:\n{model}")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test small variant
    model_small = create_vgg('small')
    output_small = model_small(x)
    print(f"\nVGG Small output shape: {output_small.shape}")
