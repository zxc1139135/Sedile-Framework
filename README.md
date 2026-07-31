# Sedile

This package implements the finite-field, fixed-point, harmonic-code, share conversion, secure-operator, availability, model, and federated-training components.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[full,test]"
pytest -q
```

## Download the vision datasets

```bash
harmobridge download-data --dataset mnist --output artifacts/data
harmobridge download-data --dataset cifar10 --output artifacts/data
harmobridge download-data --dataset cifar100 --output artifacts/data
```

The command uses the official torchvision dataset loaders and verifies the resulting train/test sizes through the loader.

## Paper-mode federated training

The production runner fixes the reported experiment design:

- 20 data owners;
- 16 admitted owners per round;
- one batch of 128 examples per admitted owner;
- Dirichlet class skew with `alpha=0.3`;
- synchronous FedSGD for 500 rounds;
- momentum 0.9 and weight decay `5e-4`.

The protocol-fidelity path uses SmallCNN on MNIST. The utility experiments reported in the main accuracy table use these fixed pairs:

- VGG-11 without batch normalization on CIFAR-10;
- VGG-11 without batch normalization on CIFAR-100;
- Fixup ResNet-20 on CIFAR-10;
- BatchNorm ResNet-20 on CIFAR-100.

Example:

```bash
harmobridge train \
  --dataset cifar10 \
  --model fixup_resnet20 \
  --learning-rate <RECORDED_VALUE> \
  --data-root artifacts/data \
  --output outputs/cifar10-fixup \
  --device cuda
```

The runner saves `checkpoint.pt` every round, supports `--resume`, appends test metrics at the configured interval, and writes `summary.json` at completion.

## Protocol validation path

By default, aggregated gradients pass through a 61-bit-prime, 16-fractional-bit encode/share/reconstruct/decode round trip before the server update. This checks the fixed-point and additive-share representation used by the protocol simulator. `--plaintext-aggregate` disables this validation path for a baseline run.

The local simulator validates algebraic correctness and the paper's bridge/operator design.

Availability calculations:

```bash
harmobridge availability --omission 0.10
```
