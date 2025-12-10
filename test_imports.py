"""
Test script to verify all imports work correctly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

# Test config imports
try:
    from config import get_config, ExperimentConfig
    print("[OK] config module")
except Exception as e:
    print(f"[FAIL] config module: {e}")

# Test crypto imports
try:
    from crypto import (
        generate_keypair_fast,
        ShamirSecretSharing,
        HarmonicEncoder
    )
    print("[OK] crypto module")
except Exception as e:
    print(f"[FAIL] crypto module: {e}")

# Test data imports
try:
    from data import FederatedDataLoader, DirichletPartitioner
    print("[OK] data module")
except Exception as e:
    print(f"[FAIL] data module: {e}")

# Test models imports
try:
    from models import get_model, create_mlp, create_lenet, create_vgg, create_resnet
    print("[OK] models module")
except Exception as e:
    print(f"[FAIL] models module: {e}")

# Test protocols imports
try:
    from protocols import (
        create_partitioner,
        PartitionDataManager,
        DistributedTrainingOrchestrator
    )
    print("[OK] protocols module")
except Exception as e:
    print(f"[FAIL] protocols module: {e}")

# Test utils imports
try:
    from utils import set_seed, MetricsTracker, ResultsSaver
    print("[OK] utils module")
except Exception as e:
    print(f"[FAIL] utils module: {e}")

print("\nAll imports tested successfully!")
