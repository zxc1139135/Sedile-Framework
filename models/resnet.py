"""
ResNet Model Implementation.

ResNet-18 architecture adapted for SVHN (32x32 images).
Used for SVHN dataset as specified in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Optional


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.
    """
    expansion = 1
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        """
        Initialize BasicBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution
            downsample: Downsampling layer for skip connection
        """
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet-50/101/152.
    """
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet architecture for CIFAR/SVHN (32x32 images).
    
    Modified from standard ResNet for smaller image sizes.
    """
    
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        input_channels: int = 3,
        num_classes: int = 10
    ):
        """
        Initialize ResNet.
        
        Args:
            block: Block type (BasicBlock or Bottleneck)
            layers: Number of blocks in each layer
            input_channels: Number of input channels
            num_classes: Number of output classes
        """
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution (modified for 32x32 input)
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Weight initialization
        self._initialize_weights()
    
    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Build a residual layer."""
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet18(input_channels: int = 3, num_classes: int = 10) -> ResNet:
    """Create ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], input_channels, num_classes)


def resnet34(input_channels: int = 3, num_classes: int = 10) -> ResNet:
    """Create ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], input_channels, num_classes)


def resnet50(input_channels: int = 3, num_classes: int = 10) -> ResNet:
    """Create ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], input_channels, num_classes)


def create_resnet(
    arch: str = 'resnet18',
    input_channels: int = 3,
    num_classes: int = 10
) -> nn.Module:
    """
    Factory function to create ResNet model.
    
    Args:
        arch: ResNet variant ('resnet18', 'resnet34', 'resnet50')
        input_channels: Number of input channels
        num_classes: Number of output classes
        
    Returns:
        ResNet model instance
    """
    models = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50
    }
    
    if arch not in models:
        raise ValueError(f"Unknown ResNet architecture: {arch}")
    
    return models[arch](input_channels, num_classes)


if __name__ == '__main__':
    # Test ResNet-18
    model = create_resnet('resnet18')
    print(f"Model architecture:\n{model}")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
