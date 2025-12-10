#!/usr/bin/env python3
"""
Test script for Sedile framework.

Runs basic tests on all components to verify correct implementation.
"""

import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_crypto_paillier():
    """Test Paillier encryption."""
    print("Testing Paillier encryption...")
    
    from crypto.paillier import generate_keypair_fast
    
    keys = generate_keypair_fast(512)
    pk, sk = keys.public, keys.private
    
    # Test encryption/decryption
    m = 42
    c = pk.encrypt(m)
    decrypted = sk.decrypt(c)
    assert decrypted == m, f"Decryption failed: {decrypted} != {m}"
    
    # Test homomorphic addition
    m1, m2 = 100, 200
    c1, c2 = pk.encrypt(m1), pk.encrypt(m2)
    c_sum = pk.add(c1, c2)
    assert sk.decrypt(c_sum) == m1 + m2, "Homomorphic addition failed"
    
    # Test scalar multiplication
    scalar = 5
    c_mult = pk.scalar_mult(c1, scalar)
    assert sk.decrypt(c_mult) == m1 * scalar, "Scalar multiplication failed"
    
    print("  Paillier tests passed!")
    return True


def test_crypto_shamir():
    """Test Shamir secret sharing."""
    print("Testing Shamir secret sharing...")
    
    from crypto.shamir import ShamirSecretSharing
    
    prime = 2**27 + 1
    ss = ShamirSecretSharing(threshold=3, num_shares=10, prime=prime)
    
    # Test secret sharing
    secret = 12345
    shares = ss.share_secret(secret)
    reconstructed = ss.reconstruct_secret(shares[:3])
    assert reconstructed == secret, f"Reconstruction failed: {reconstructed} != {secret}"
    
    # Test with different subset
    reconstructed2 = ss.reconstruct_secret(shares[5:8])
    assert reconstructed2 == secret, "Reconstruction with different shares failed"
    
    # Test matrix sharing
    matrix = np.array([[1, 2], [3, 4]], dtype=np.int64)
    share_matrices = ss.share_matrix(matrix)
    reconstructed_matrix = ss.reconstruct_matrix(share_matrices[:3])
    assert np.array_equal(matrix, reconstructed_matrix), "Matrix reconstruction failed"
    
    print("  Shamir tests passed!")
    return True


def test_crypto_harmonic():
    """Test Harmonic coding."""
    print("Testing Harmonic coding...")
    
    from crypto.harmonic import HarmonicCoder, create_harmonic_system
    
    prime = 2**27 + 1
    
    # Test HarmonicCoder
    coder = HarmonicCoder(num_partitions=3, degree=3, prime=prime)
    
    data = np.random.randint(0, 1000, size=(10, 5), dtype=np.int64)
    mask = np.random.randint(0, prime, size=data.shape, dtype=np.int64)
    
    encoded, _ = coder.encode(data, group_id=1, client_idx_in_group=0, mask_Z=mask)
    assert encoded.shape == data.shape, "Encoding shape mismatch"
    
    # Test HarmonicEncoder
    encoder = create_harmonic_system(num_clients=10, num_partitions=3)
    
    full_data = np.random.randint(0, 1000, size=(30, 5), dtype=np.int64)
    partitions = encoder.partition_data(full_data)
    assert len(partitions) == encoder.K, "Partition count mismatch"
    
    print("  Harmonic coding tests passed!")
    return True


def test_data_loading():
    """Test data loading and partitioning."""
    print("Testing data loading...")
    
    from data import FederatedDataLoader
    
    # Test with small setup
    loader = FederatedDataLoader(
        dataset_name='mnist',
        num_clients=10,
        distribution_type='dirichlet',
        distribution_param=0.5,
        batch_size=32
    )
    
    assert loader.num_clients == 10, "Client count mismatch"
    
    # Test client loader
    client_loader = loader.get_client_loader(0)
    batch = next(iter(client_loader))
    assert len(batch) == 2, "Batch should have data and labels"
    assert batch[0].shape[1:] == (1, 28, 28), "Data shape mismatch"
    
    # Test distributions
    dist = loader.get_client_distribution(0)
    assert len(dist) == 10, "Distribution should have 10 classes"
    assert np.isclose(dist.sum(), 1.0), "Distribution should sum to 1"
    
    print("  Data loading tests passed!")
    return True


def test_models():
    """Test model creation."""
    print("Testing models...")
    
    from models import get_model
    
    datasets_shapes = {
        'mnist': (1, 28, 28),
        'fmnist': (1, 28, 28),
        'cifar10': (3, 32, 32),
        'svhn': (3, 32, 32)
    }
    
    for dataset, shape in datasets_shapes.items():
        model = get_model(dataset)
        x = torch.randn(2, *shape)
        output = model(x)
        assert output.shape == (2, 10), f"{dataset} model output shape mismatch"
        print(f"  {dataset}: OK")
    
    print("  Model tests passed!")
    return True


def test_partitioning():
    """Test client partitioning."""
    print("Testing client partitioning...")
    
    from protocols import create_partitioner
    
    num_clients = 20
    num_partitions = 4
    num_classes = 10
    
    # Generate random distributions
    distributions = np.random.dirichlet([0.5] * num_classes, size=num_clients)
    
    partitioner = create_partitioner(
        num_clients=num_clients,
        num_partitions=num_partitions,
        metric='euclidean'
    )
    
    partitions = partitioner.partition(distributions)
    
    # Verify all clients assigned
    all_clients = set()
    for p in partitions:
        all_clients.update(p)
    assert len(all_clients) == num_clients, "Not all clients assigned"
    
    # Verify partition count
    assert len(partitions) == num_partitions, "Wrong number of partitions"
    
    print("  Partitioning tests passed!")
    return True


def test_training():
    """Test training components."""
    print("Testing training components...")
    
    from models import get_model
    from protocols import LocalTrainer
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create simple model and data
    model = get_model('mnist')
    
    data = torch.randn(100, 1, 28, 28)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=32)
    
    # Test local trainer
    trainer = LocalTrainer(model, learning_rate=0.01, device='cpu')
    
    # Train for one epoch
    loss, num_samples = trainer.train_epoch(loader, num_epochs=1)
    assert loss >= 0, "Loss should be non-negative"
    assert num_samples == 100, "Sample count mismatch"
    
    # Test evaluation
    test_loss, accuracy = trainer.evaluate(loader)
    assert 0 <= accuracy <= 100, "Accuracy should be between 0 and 100"
    
    print("  Training tests passed!")
    return True


def test_utils():
    """Test utility functions."""
    print("Testing utilities...")
    
    from utils import set_seed, MetricsTracker, get_device, format_time
    import time
    
    # Test seed setup
    set_seed(42)
    r1 = np.random.rand()
    set_seed(42)
    r2 = np.random.rand()
    assert r1 == r2, "Seed setup failed"
    
    # Test metric tracker
    tracker = MetricsTracker()
    for i in range(10):
        tracker.add_scalar('accuracy', 50 + i * 5, i)
    assert tracker.get_best('accuracy') == 95, "Best metric tracking failed"
    assert tracker.get_latest('accuracy') == 95, "Latest metric tracking failed"
    
    # Test device detection
    device = get_device()
    assert device in ['cuda', 'cpu'], "Invalid device"
    
    # Test time formatting
    formatted = format_time(3661)
    assert 'h' in formatted or 'm' in formatted, "Time formatting failed"
    
    print("  Utility tests passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running Sedile Framework Tests")
    print("="*60)
    
    tests = [
        ("Paillier Encryption", test_crypto_paillier),
        ("Shamir Secret Sharing", test_crypto_shamir),
        ("Harmonic Coding", test_crypto_harmonic),
        ("Data Loading", test_data_loading),
        ("Models", test_models),
        ("Client Partitioning", test_partitioning),
        ("Training", test_training),
        ("Utilities", test_utils),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
