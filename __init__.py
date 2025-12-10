"""
Sedile: Privacy-Preserving Distributed Deep Learning Framework

A scalable framework for privacy-preserving distributed deep learning
that provides dual-threat protection against both honest-but-curious
servers and colluding clients under non-IID data settings.

Main components:
- Similarity-driven client partitioning (Protocol 1)
- Intra-partition data sharing with secret sharing and harmonic coding
- Inter-partition model training with secure aggregation

Supported datasets:
- MNIST (MLP)
- Fashion-MNIST (LeNet-5)
- CIFAR-10 (VGG)
- SVHN (ResNet-18)

Supported non-IID distributions:
- Dirichlet distribution (alpha parameter)
- Pathological distribution (kappa parameter)
"""

__version__ = '1.0.0'
__author__ = 'Anonymous'

from . import config
from . import crypto
from . import data
from . import models
from . import protocols
from . import utils
from . import experiments
