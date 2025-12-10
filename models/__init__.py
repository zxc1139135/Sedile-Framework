"""
Neural network models for Sedile framework.

Model-dataset mapping as specified in the paper:
- MNIST: MLP with two hidden layers (256 nodes each)
- Fashion-MNIST: LeNet-5
- CIFAR-10: VGG
- SVHN: ResNet-18
"""

import torch.nn as nn

from .mlp import MLP, create_mlp
from .lenet import LeNet5, LeNet5BN, create_lenet
from .vgg import VGG, VGGSmall, create_vgg
from .resnet import ResNet, BasicBlock, Bottleneck, create_resnet, resnet18, resnet34, resnet50


__all__ = [
    'MLP', 'create_mlp',
    'LeNet5', 'LeNet5BN', 'create_lenet',
    'VGG', 'VGGSmall', 'create_vgg',
    'ResNet', 'BasicBlock', 'Bottleneck', 'create_resnet',
    'resnet18', 'resnet34', 'resnet50',
    'get_model'
]


# Dataset to model mapping
DATASET_MODEL_MAP = {
    'mnist': ('mlp', {'input_channels': 1, 'input_size': 28}),
    'fmnist': ('lenet', {'input_channels': 1}),
    'cifar10': ('vgg', {'input_channels': 3}),
    'svhn': ('resnet18', {'input_channels': 3})
}


def get_model(
    dataset_name: str,
    num_classes: int = 10,
    **kwargs
) -> nn.Module:
    """
    Get appropriate model for a dataset.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'fmnist', 'cifar10', 'svhn')
        num_classes: Number of output classes
        **kwargs: Additional model arguments
        
    Returns:
        Model instance appropriate for the dataset
    """
    if dataset_name not in DATASET_MODEL_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    model_type, default_kwargs = DATASET_MODEL_MAP[dataset_name]
    model_kwargs = {**default_kwargs, 'num_classes': num_classes, **kwargs}
    
    if model_type == 'mlp':
        return create_mlp(**model_kwargs)
    elif model_type == 'lenet':
        return create_lenet(**model_kwargs)
    elif model_type == 'vgg':
        return create_vgg(arch='vgg11', **model_kwargs)
    elif model_type == 'resnet18':
        return create_resnet(arch='resnet18', **model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model factory
    for dataset in ['mnist', 'fmnist', 'cifar10', 'svhn']:
        model = get_model(dataset)
        params = count_parameters(model)
        print(f"{dataset}: {type(model).__name__} with {params:,} parameters")
