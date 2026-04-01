from __future__ import annotations

import argparse

from common import PROJECT_ROOT
from torch_common import (
    DEFAULT_MAE_CHECKPOINT,
    add_torch_args,
    build_training_config,
    ensure_mae_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain TacVerse ViT-S/16 with MAE.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="MAE pretraining image size. Keep this aligned with ViT fine-tuning.",
    )
    add_torch_args(parser)
    parser.set_defaults(
        mae_checkpoint=DEFAULT_MAE_CHECKPOINT,
        batch_size=32,
        epochs=10,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_training_config(args)
    checkpoint_path = ensure_mae_checkpoint(config)
    if checkpoint_path.is_relative_to(PROJECT_ROOT):
        display_path = checkpoint_path.relative_to(PROJECT_ROOT)
    else:
        display_path = checkpoint_path
    print(f"Saved TacVerse MAE checkpoint to {display_path}")


if __name__ == "__main__":
    main()
