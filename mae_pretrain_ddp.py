"""TacVerse MAE pretraining with multi-GPU DDP support.

Usage:
    # Single GPU
    python code/mae_pretrain_ddp.py --image-size 224 --mae-epochs 100

    # 4-GPU DDP
    torchrun --nproc_per_node=4 code/mae_pretrain_ddp.py --image-size 224 --mae-epochs 100
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch_common import (
    DEFAULT_MAE_CHECKPOINT,
    ImageOnlyDataset,
    MaskedAutoencoderViT,
    add_torch_args,
    build_training_config,
    collect_tacverse_image_paths,
    mae_signature,
    set_seed,
    validate_mae_checkpoint,
    _param_groups_weight_decay,
    _cosine_schedule_with_warmup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain TacVerse ViT-S/16 with MAE (DDP).")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="MAE pretraining image size.",
    )
    parser.add_argument(
        "--mae-from-imagenet",
        action="store_true",
        help="Initialize MAE encoder from ImageNet pretrained weights instead of random.",
    )
    add_torch_args(parser)
    parser.set_defaults(
        mae_checkpoint=DEFAULT_MAE_CHECKPOINT,
        batch_size=128,
        epochs=30,
    )
    return parser.parse_args()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def setup_ddp() -> torch.device:
    if "RANK" not in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")
        return torch.device("cuda")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device(f"cuda:{local_rank}")


def cleanup_ddp() -> None:
    if is_dist():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    config = build_training_config(args)

    checkpoint_path = config.mae_checkpoint
    if args.mae_from_imagenet:
        checkpoint_path = checkpoint_path.parent / checkpoint_path.name.replace(".pth", "_from_imagenet.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    device = setup_ddp()
    world_size = get_world_size()
    rank = get_rank()

    if checkpoint_path.exists():
        validate_mae_checkpoint(checkpoint_path, config)
        if is_main():
            print(f"MAE checkpoint already exists: {checkpoint_path}")
        cleanup_ddp()
        return

    set_seed(config.seed + rank)

    image_paths = collect_tacverse_image_paths(train_only=True)
    if not image_paths:
        raise RuntimeError("No TacVerse images found for MAE pretraining.")
    if config.mae_max_images > 0:
        image_paths = image_paths[: config.mae_max_images]

    dataset = ImageOnlyDataset(image_paths, image_size=config.image_size)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_dist() else None
    loader = DataLoader(
        dataset,
        batch_size=config.mae_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )

    model = MaskedAutoencoderViT(
        image_size=config.image_size,
        mask_ratio=config.mae_mask_ratio,
        pretrained_encoder=args.mae_from_imagenet,
    ).to(device)

    if is_dist():
        model = DDP(model, device_ids=[device.index])
    raw_model = model.module if is_dist() else model

    param_groups = _param_groups_weight_decay(model, config.mae_weight_decay, config.mae_lr)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    total_steps = config.mae_epochs * len(loader)
    warmup_steps = min(len(loader), total_steps // 5)
    scheduler = _cosine_schedule_with_warmup(optimizer, total_steps, warmup_steps)

    if is_main():
        print(
            f"MAE DDP pretraining: {len(dataset)} images, "
            f"{world_size} GPUs, batch {config.mae_batch_size}×{world_size}="
            f"{config.mae_batch_size * world_size}, "
            f"{config.mae_epochs} epochs, {len(loader)} steps/epoch"
        )

    model.train()
    for epoch in range(1, config.mae_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        running_loss = 0.0
        total_samples = 0
        for images in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        if is_dist():
            stats = torch.tensor([running_loss, float(total_samples)], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            running_loss = float(stats[0])
            total_samples = int(stats[1])

        if is_main():
            mean_loss = running_loss / max(1, total_samples)
            lr = optimizer.param_groups[0]["lr"]
            print(f"MAE epoch {epoch}/{config.mae_epochs}: loss={mean_loss:.6f}  lr={lr:.2e}")

    if is_main():
        checkpoint = {
            "encoder_state_dict": raw_model.encoder.state_dict(),
            "num_images": len(dataset),
            **mae_signature(config),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved MAE checkpoint: {checkpoint_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
