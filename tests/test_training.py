import numpy as np
import pytest
import torch
from torch import nn

from harmobridge.federated import fedsgd_step
from harmobridge.training import TrainingConfig, build_model, secure_gradient_roundtrip, stable_dirichlet_partition


def test_paper_config_requires_learning_rate():
    with pytest.raises(ValueError, match="learning rate"):
        TrainingConfig(dataset="mnist", model="smallcnn", learning_rate=None).validate()


def test_model_factory_shapes():
    assert build_model("smallcnn", "mnist")(torch.randn(2, 1, 28, 28)).shape == (2, 10)
    assert build_model("resnet20_batchnorm", "cifar100")(torch.randn(2, 3, 32, 32)).shape == (2, 100)


def test_stable_partition_has_all_clients():
    labels = np.repeat(np.arange(10), 100)
    parts = stable_dirichlet_partition(labels, clients=20, alpha=0.3, seed=3)
    assert len(parts) == 20
    assert min(map(len, parts)) > 0
    assert sum(map(len, parts)) == len(labels)


def test_secure_gradient_roundtrip_and_fedsgd():
    model = nn.Linear(2, 2)
    before = model.weight.detach().clone()
    batches = [(torch.tensor([[1.0, 0.0]]), torch.tensor([0])), (torch.tensor([[0.0, 1.0]]), torch.tensor([1]))]
    fedsgd_step(
        model,
        batches,
        [0.5, 0.5],
        learning_rate=0.1,
        momentum_buffers={},
        momentum=0.9,
        weight_decay=0.0,
        gradient_transform=secure_gradient_roundtrip(seed=4),
    )
    assert not torch.equal(before, model.weight)


def test_client_batch_stream_state_roundtrip():
    from torch.utils.data import TensorDataset
    from harmobridge.training import ClientBatchStream

    dataset = TensorDataset(torch.arange(20).float().unsqueeze(1), torch.arange(20))
    left = ClientBatchStream(dataset, np.arange(20), batch_size=4, seed=9)
    left.next()
    state = left.state_dict()
    expected = left.next()[1]
    right = ClientBatchStream(dataset, np.arange(20), batch_size=4, seed=9)
    right.load_state_dict(state)
    actual = right.next()[1]
    assert torch.equal(expected, actual)


def test_model_factory_rejects_nonpaper_workload():
    with pytest.raises(ValueError, match="Unsupported paper workload"):
        build_model("fixup_resnet20", "cifar100")
