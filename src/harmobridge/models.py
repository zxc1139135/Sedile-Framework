from __future__ import annotations

import math

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, channels: int = 1, classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(32 * 4 * 4, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class VGG11NoBN(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        layout = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        layers: list[nn.Module] = []
        channels = 3
        for item in layout:
            if item == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.extend([nn.Conv2d(channels, int(item), 3, padding=1), nn.ReLU(inplace=False)])
                channels = int(item)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class ResidualBlock(nn.Module):
    """CIFAR basic block used by the 20-layer residual networks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        batch_norm: bool = True,
        fixup: bool = False,
        block_count: int = 9,
    ) -> None:
        super().__init__()
        if fixup and batch_norm:
            raise ValueError("Fixup blocks must not use batch normalization")

        self.fixup = fixup
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=not batch_norm,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=not batch_norm)
        self.norm1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.norm2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.shortcut = self._make_shortcut(in_channels, out_channels, stride)

        if fixup:
            self.bias1a = nn.Parameter(torch.zeros(1))
            self.bias1b = nn.Parameter(torch.zeros(1))
            self.bias2a = nn.Parameter(torch.zeros(1))
            self.bias2b = nn.Parameter(torch.zeros(1))
            self.scale = nn.Parameter(torch.ones(1))
            nn.init.normal_(self.conv1.weight, mean=0.0, std=math.sqrt(2.0 / self.conv1.weight[0].numel()) * block_count ** -0.5)
            nn.init.zeros_(self.conv2.weight)
        else:
            self.register_parameter("bias1a", None)
            self.register_parameter("bias1b", None)
            self.register_parameter("bias2a", None)
            self.register_parameter("bias2b", None)
            self.register_parameter("scale", None)

    @staticmethod
    def _make_shortcut(in_channels: int, out_channels: int, stride: int) -> nn.Module:
        if stride == 1 and in_channels == out_channels:
            return nn.Identity()
        return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        value = x
        if self.fixup:
            value = value + self.bias1a
        value = self.conv1(value)
        if self.fixup:
            value = value + self.bias1b
        value = torch.relu(self.norm1(value))
        if self.fixup:
            value = value + self.bias2a
        value = self.conv2(value)
        value = self.norm2(value)
        if self.fixup:
            value = value * self.scale + self.bias2b
        return torch.relu(value + residual)


class ResNet20(nn.Module):
    """Canonical CIFAR ResNet-20 with three blocks per stage."""

    def __init__(self, classes: int = 10, batch_norm: bool = True, fixup: bool = False) -> None:
        super().__init__()
        if fixup and batch_norm:
            raise ValueError("Fixup ResNet-20 must be created with batch_norm=False")

        self.fixup = fixup
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=not batch_norm)
        self.stem_norm = nn.BatchNorm2d(16) if batch_norm else nn.Identity()
        self.stem_bias = nn.Parameter(torch.zeros(1)) if fixup else None
        self.blocks = nn.Sequential(
            *self._stage(16, 16, 3, stride=1, batch_norm=batch_norm, fixup=fixup),
            *self._stage(16, 32, 3, stride=2, batch_norm=batch_norm, fixup=fixup),
            *self._stage(32, 64, 3, stride=2, batch_norm=batch_norm, fixup=fixup),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, classes)
        self.classifier_bias = nn.Parameter(torch.zeros(1)) if fixup else None
        self._initialize(batch_norm=batch_norm, fixup=fixup)

    @staticmethod
    def _stage(
        in_channels: int,
        out_channels: int,
        count: int,
        stride: int,
        batch_norm: bool,
        fixup: bool,
    ) -> list[nn.Module]:
        blocks: list[nn.Module] = [
            ResidualBlock(in_channels, out_channels, stride, batch_norm, fixup)
        ]
        blocks.extend(
            ResidualBlock(out_channels, out_channels, 1, batch_norm, fixup)
            for _ in range(count - 1)
        )
        return blocks

    def _initialize(self, batch_norm: bool, fixup: bool) -> None:
        if not fixup:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif batch_norm and isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        else:
            nn.init.normal_(self.stem.weight, mean=0.0, std=math.sqrt(2.0 / self.stem.weight[0].numel()))
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.stem_norm(self.stem(x))
        if self.fixup:
            value = value + self.stem_bias
        value = torch.relu(value)
        value = self.pool(self.blocks(value)).flatten(1)
        if self.fixup:
            value = value + self.classifier_bias
        return self.classifier(value)


def fixup_resnet20(classes: int = 10) -> ResNet20:
    return ResNet20(classes=classes, batch_norm=False, fixup=True)


def resnet20_batchnorm(classes: int = 10) -> ResNet20:
    return ResNet20(classes=classes, batch_norm=True, fixup=False)
