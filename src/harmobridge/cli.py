from __future__ import annotations

import argparse
import json

from .availability import committee_success, dual_server_success, replicated_hc_success, unreplicated_success
from .datasets import load_vision_datasets
from .training import TrainingConfig, run_federated_training


def availability_command(args: argparse.Namespace) -> None:
    output = {
        "unreplicated": unreplicated_success(args.omission),
        "replicated_hc": replicated_hc_success(args.omission),
        "dual_server": dual_server_success(args.omission),
        "committee": committee_success(args.omission),
    }
    print(json.dumps(output, indent=2))


def download_command(args: argparse.Namespace) -> None:
    train, test = load_vision_datasets(args.dataset, args.output, download=True, augmentation=False)
    print(json.dumps({"dataset": args.dataset, "train": len(train), "test": len(test), "root": args.output}, indent=2))


def train_command(args: argparse.Namespace) -> None:
    config = TrainingConfig(
        dataset=args.dataset,
        model=args.model,
        learning_rate=args.learning_rate,
        rounds=500,
        owners=20,
        admitted_per_round=16,
        batch_size=128,
        dirichlet_alpha=0.3,
        momentum=0.9,
        weight_decay=5e-4,
        seed=args.seed,
        evaluation_interval=args.evaluation_interval,
        augmentation=args.augmentation,
        secure_roundtrip=not args.plaintext_aggregate,
        fractional_bits=16,
    )
    summary = run_federated_training(config, args.data_root, args.output, device=args.device, resume=args.resume)
    print(json.dumps(summary["final"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HarmoBridge paper experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    availability = sub.add_parser("availability")
    availability.add_argument("--omission", type=float, required=True)
    availability.set_defaults(func=availability_command)

    download = sub.add_parser("download-data")
    download.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"], required=True)
    download.add_argument("--output", default="artifacts/data")
    download.set_defaults(func=download_command)

    train = sub.add_parser("train")
    train.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"], required=True)
    train.add_argument("--model", choices=["smallcnn", "vgg11_no_batchnorm", "fixup_resnet20", "resnet20_batchnorm"], required=True)
    train.add_argument("--learning-rate", type=float, required=True, help="Required because the manuscript does not state this value")
    train.add_argument("--data-root", default="artifacts/data")
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--evaluation-interval", type=int, default=10)
    train.add_argument("--augmentation", action="store_true", help="Explicit implementation choice; disabled by default")
    train.add_argument("--plaintext-aggregate", action="store_true", help="Disable the fixed-point share round-trip verifier")
    train.add_argument("--resume")
    train.set_defaults(func=train_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
