from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all TacVerse benchmark scripts.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize images to a square of this size.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="resnet18,convnext_tiny,vit_small_patch16_224",
        help="Comma-separated backbones.",
    )
    parser.add_argument(
        "--vit-init-modes",
        type=str,
        default="random,imagenet,mae",
        help="Comma-separated ViT init modes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size for supervised training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of supervised training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for supervised fine-tuning.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for supervised fine-tuning.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early-stopping patience.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto,cuda,mps. CPU training is disabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in each experiment.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated seed list. Overrides --seed when provided.",
    )
    parser.add_argument(
        "--allow-random-fallback",
        action="store_true",
        help="Fallback to random init if ImageNet weights are unavailable offline.",
    )
    parser.add_argument(
        "--resnet-imagenet-weights",
        type=Path,
        default=None,
        help="Optional local path to ResNet-18 ImageNet weights.",
    )
    parser.add_argument(
        "--convnext-imagenet-weights",
        type=Path,
        default=None,
        help="Optional local path to ConvNeXt-Tiny ImageNet weights.",
    )
    parser.add_argument(
        "--vit-imagenet-weights",
        type=Path,
        default=None,
        help="Optional local path to ViT-S/16 ImageNet weights.",
    )
    parser.add_argument(
        "--mae-checkpoint",
        type=Path,
        default=None,
        help="Where to save or load the TacVerse MAE ViT checkpoint. "
        "If not set, each task script uses its own default.",
    )
    parser.add_argument(
        "--mae-epochs",
        type=int,
        default=100,
        help="Epochs used when pretraining TacVerse MAE.",
    )
    parser.add_argument(
        "--mae-batch-size",
        type=int,
        default=256,
        help="Mini-batch size for TacVerse MAE pretraining.",
    )
    parser.add_argument(
        "--mae-lr",
        type=float,
        default=1.5e-4,
        help="Learning rate for TacVerse MAE pretraining.",
    )
    parser.add_argument(
        "--mae-weight-decay",
        type=float,
        default=0.05,
        help="Weight decay for TacVerse MAE pretraining.",
    )
    parser.add_argument(
        "--mae-mask-ratio",
        type=float,
        default=0.75,
        help="Patch masking ratio used for TacVerse MAE pretraining.",
    )
    parser.add_argument(
        "--mae-max-images",
        type=int,
        default=0,
        help="Optional cap on the number of TacVerse images used during MAE pretraining.",
    )
    return parser.parse_args()


def add_common_args(command: list[str], args: argparse.Namespace) -> list[str]:
    command.extend(
        [
            "--image-size",
            str(args.image_size),
            "--models",
            args.models,
            "--vit-init-modes",
            args.vit_init_modes,
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--patience",
            str(args.patience),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--seeds",
            args.seeds,
            "--mae-epochs",
            str(args.mae_epochs),
            "--mae-batch-size",
            str(args.mae_batch_size),
            "--mae-lr",
            str(args.mae_lr),
            "--mae-weight-decay",
            str(args.mae_weight_decay),
            "--mae-mask-ratio",
            str(args.mae_mask_ratio),
            "--mae-max-images",
            str(args.mae_max_images),
        ]
    )
    if args.mae_checkpoint is not None:
        command.extend(["--mae-checkpoint", str(args.mae_checkpoint)])
    if args.allow_random_fallback:
        command.append("--allow-random-fallback")
    if args.resnet_imagenet_weights is not None:
        command.extend(["--resnet-imagenet-weights", str(args.resnet_imagenet_weights)])
    if args.convnext_imagenet_weights is not None:
        command.extend(["--convnext-imagenet-weights", str(args.convnext_imagenet_weights)])
    if args.vit_imagenet_weights is not None:
        command.extend(["--vit-imagenet-weights", str(args.vit_imagenet_weights)])
    return command


def main() -> None:
    args = parse_args()
    scripts = [
        "force_regression.py",
        "shape_classification.py",
        "grating_classification.py",
    ]

    for script_name in scripts:
        command = [sys.executable, str(CODE_DIR / script_name)]
        command = add_common_args(command, args)
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
