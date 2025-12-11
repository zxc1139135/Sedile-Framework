# Sedile: Efficient and Privacy-Preserving Distributed Deep Learning for Non-IID Data

A scalable framework for privacy-preserving distributed deep learning that addresses both computational scalability and privacy concerns for non-IID data.

## Overview

Sedile is a novel privacy-preserving distributed learning framework that enables multiple clients to collaboratively train deep learning models while:

1. **Protecting client data privacy** against both the server and colluding clients
2. **Scaling efficiently** to large numbers of participants through strategic partitioning
3. **Maintaining model accuracy** under non-IID data distributions

### Key Features

- **Similarity-Driven Client Partitioning**: Groups clients based on data distribution similarity to optimize training convergence
- **Intra-Partition Data Sharing**: Uses Shamir secret sharing for privacy-preserving data distribution within partitions
- **Harmonic Coding**: Enables exact gradient computation on encoded data
- **Dual-Threat Protection**: Provides T-privacy against colluding adversaries

## Architecture

```
sedile/
├── config/                 # Configuration management
│   ├── __init__.py
│   └── config.py          # Experiment configurations
├── crypto/                 # Cryptographic primitives
│   ├── __init__.py
│   ├── paillier.py        # Paillier encryption
│   ├── shamir.py          # Shamir secret sharing
│   └── harmonic.py        # Harmonic coding
├── data/                   # Data loading and partitioning
│   ├── __init__.py
│   └── data_loader.py     # Non-IID data distribution
├── models/                 # Neural network architectures
│   ├── __init__.py
│   ├── mlp.py             # Multi-layer perceptron
│   ├── lenet.py           # LeNet-5
│   ├── vgg.py             # VGG
│   └── resnet.py          # ResNet-18
├── protocols/              # Privacy-preserving protocols
│   ├── __init__.py
│   ├── partitioning.py    # Client partitioning
│   ├── data_sharing.py    # Intra-partition sharing
│   └── training.py        # Distributed training
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── helpers.py         # Helper functions
├── experiments/            # Experiment runners
│   ├── __init__.py
│   └── run_all.py         # Batch experiments
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd sedile

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Single Experiment

Run a single experiment with specified parameters:

```bash
python main.py --dataset mnist \
               --num-clients 50 \
               --num-partitions 5 \
               --distribution dirichlet \
               --param 0.1 \
               --rounds 300 \
               --lr 0.015
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | mnist | Dataset: mnist, fmnist, cifar10, svhn |
| `--num-clients` | int | 50 | Number of clients |
| `--num-partitions` | int | 5 | Number of partitions |
| `--distribution` | str | dirichlet | Distribution type: dirichlet, pathological |
| `--param` | float | 0.1 | Distribution parameter |
| `--rounds` | int | 300 | Number of training rounds |
| `--local-epochs` | int | 3 | Local epochs per round |
| `--batch-size` | int | 32 | Batch size |
| `--lr` | float | 0.015 | Learning rate |
| `--metric` | str | euclidean | Similarity metric: euclidean, cosine, kl |
| `--threshold` | int | 3 | T-privacy threshold |
| `--seed` | int | 42 | Random seed |
| `--output-dir` | str | ./outputs | Output directory |
| `--device` | str | auto | Device: cuda, cpu |

### Batch Experiments

Reproduce all paper experiments:

```bash
python experiments/run_all.py --output-dir ./outputs/batch
```

Run specific datasets:

```bash
python experiments/run_all.py --datasets mnist cifar10
```

## Experimental Setup

### Datasets

| Dataset | Model | Input Size | Classes |
|---------|-------|------------|---------|
| MNIST | MLP (256-256) | 28x28x1 | 10 |
| Fashion-MNIST | LeNet-5 | 28x28x1 | 10 |
| CIFAR-10 | VGG-11 | 32x32x3 | 10 |
| SVHN | ResNet-18 | 32x32x3 | 10 |

### Data Distribution

**Dirichlet Distribution**
- alpha = 0.1: Highly non-IID (significant label skew)
- alpha = 1.0: Moderately non-IID

**Pathological Distribution**
- kappa = 2: Each client has only 2 classes
- kappa = 5: Each client has only 5 classes

### Training Configuration

- Clients: N = 50
- Partitions: V = 5
- Local epochs: E = 3
- Batch size: B = 32
- Learning rate: lr = 0.015
- Training rounds: R = 300

## Output Structure

```
outputs/
└── <dataset>_<distribution>_<param>_<timestamp>/
    ├── logs/
    │   └── sedile_<dataset>_<timestamp>.log
    ├── config.json           # Experiment configuration
    ├── training_history.json # Training metrics
    ├── final_model.pt        # Trained model weights
    ├── partitions.json       # Client partitions
    ├── client_distributions.npy
    ├── training_curves.png
    └── distribution_heatmap.png
```

## API Reference

### Main Classes

#### SedileExperiment
```python
from main import SedileExperiment

experiment = SedileExperiment(
    dataset='mnist',
    num_clients=50,
    num_partitions=5,
    distribution_type='dirichlet',
    distribution_param=0.1
)
history = experiment.run()
```

#### FederatedDataLoader
```python
from data import FederatedDataLoader

loader = FederatedDataLoader(
    dataset_name='mnist',
    num_clients=50,
    distribution_type='dirichlet',
    distribution_param=0.1,
    batch_size=32
)
```

#### ClientPartitioner
```python
from protocols import create_partitioner

partitioner = create_partitioner(
    num_clients=50,
    num_partitions=5,
    metric='euclidean'
)
partitions = partitioner.partition(distributions)
```

## License

This project is released for academic research purposes.
