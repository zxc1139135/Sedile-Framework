import torch

from harmobridge.models import SmallCNN, VGG11NoBN, fixup_resnet20, resnet20_batchnorm


def test_paper_models_produce_expected_shapes():
    assert SmallCNN()(torch.randn(2, 1, 28, 28)).shape == (2, 10)
    assert VGG11NoBN(classes=100)(torch.randn(2, 3, 32, 32)).shape == (2, 100)
    assert fixup_resnet20(classes=10)(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert resnet20_batchnorm(classes=100)(torch.randn(2, 3, 32, 32)).shape == (2, 100)


def test_resnet20_has_nine_residual_blocks():
    assert len(fixup_resnet20().blocks) == 9
    assert len(resnet20_batchnorm().blocks) == 9
