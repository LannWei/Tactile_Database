from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from timm.models.vision_transformer import Block
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import PROJECT_ROOT, RESULTS_DIR


SUPPORTED_MODELS = ("resnet18", "convnext_tiny", "vit_small_patch16_224")
DEFAULT_MODELS = ",".join(SUPPORTED_MODELS)
DEFAULT_VIT_INITS = "random,imagenet,mae"
VIT_MODEL_NAME = "vit_small_patch16_224"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
MAE_DIR = CHECKPOINTS_DIR / "mae"

_BYTES_CACHE: dict[str, bytes] = {}


def _load_image_cached(path: str) -> Image.Image:
    if path not in _BYTES_CACHE:
        with open(path, "rb") as f:
            _BYTES_CACHE[path] = f.read()
    import io
    img = Image.open(io.BytesIO(_BYTES_CACHE[path]))
    return ImageOps.exif_transpose(img).convert("RGB")


def prefetch_images(paths: Sequence[str]) -> None:
    """Pre-load raw file bytes into memory to avoid NFS latency during training."""
    uncached = [p for p in paths if p not in _BYTES_CACHE]
    if not uncached:
        return
    total_mb = 0.0
    print(f"Pre-caching {len(uncached)} images ...", end="", flush=True)
    for i, p in enumerate(uncached):
        with open(p, "rb") as f:
            data = f.read()
        _BYTES_CACHE[p] = data
        total_mb += len(data) / 1024 / 1024
        if (i + 1) % 2000 == 0:
            print(f" {i + 1}", end="", flush=True)
    print(f" done ({total_mb:.0f} MB).", flush=True)


def format_float_tag(value: float) -> str:
    return str(value).replace(".", "p")


def default_mae_checkpoint_path(
    image_size: int,
    mask_ratio: float,
    mae_epochs: int,
    mae_max_images: int,
) -> Path:
    subset_tag = "all" if mae_max_images <= 0 else f"n{mae_max_images}"
    return MAE_DIR / (
        f"tacverse_vit_small_patch16_224_mae_img{image_size}"
        f"_mask{format_float_tag(mask_ratio)}_ep{mae_epochs}_{subset_tag}.pth"
    )


DEFAULT_MAE_CHECKPOINT = default_mae_checkpoint_path(
    image_size=224,
    mask_ratio=0.75,
    mae_epochs=100,
    mae_max_images=0,
)


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    init_mode: str


@dataclass(frozen=True)
class TrainingConfig:
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    num_workers: int
    device: str
    seed: int
    allow_random_fallback: bool
    resnet_imagenet_weights: Path | None
    convnext_imagenet_weights: Path | None
    vit_imagenet_weights: Path | None
    mae_checkpoint: Path
    mae_epochs: int
    mae_batch_size: int
    mae_lr: float
    mae_weight_decay: float
    mae_mask_ratio: float
    mae_max_images: int


@dataclass(frozen=True)
class RegressionTargetNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, target_cols: Sequence[str]) -> "RegressionTargetNormalizer":
        values = frame.loc[:, list(target_cols)].to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform_array(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def transform_frame(self, frame: pd.DataFrame, target_cols: Sequence[str]) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.loc[:, list(target_cols)] = self.transform_array(
            normalized.loc[:, list(target_cols)].to_numpy(dtype=np.float32)
        )
        return normalized


class TacverseDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
        task_type: str,
        label_col: str | None = None,
        target_cols: Sequence[str] | None = None,
        label_to_index: dict[str, int] | None = None,
        normalize: bool = True,
        is_training: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.task_type = task_type
        self.label_col = label_col
        self.target_cols = tuple(target_cols or ())
        self.label_to_index = label_to_index or {}
        if is_training:
            transform_ops: list = [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transform_ops = [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        if normalize:
            transform_ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        self.transform = transforms.Compose(transform_ops)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = _load_image_cached(row["image_path"])
        tensor = self.transform(image)

        if self.task_type == "classification":
            label = self.label_to_index[str(row[self.label_col])]
            return tensor, torch.tensor(label, dtype=torch.long)

        target = torch.tensor(
            row.loc[list(self.target_cols)].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        return tensor, target


class ImageOnlyDataset(Dataset):
    def __init__(self, image_paths: Sequence[str | Path], image_size: int) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.2, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = _load_image_cached(str(self.image_paths[index]))
        return self.transform(image)


class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        mask_ratio: float,
        decoder_embed_dim: int = 384,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        norm_pix_loss: bool = True,
        pretrained_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            VIT_MODEL_NAME,
            pretrained=pretrained_encoder,
            img_size=image_size,
            num_classes=0,
        )
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        embed_dim = self.encoder.embed_dim
        self.patch_size = int(self.encoder.patch_embed.patch_size[0])
        self.num_patches = int(self.encoder.patch_embed.num_patches)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.patch_size * self.patch_size * 3,
            bias=True,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.zeros_(self.decoder_embed.bias)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.zeros_(self.decoder_pred.bias)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        patch_size = self.patch_size
        batch_size, channels, height, width = images.shape
        if height != width or height % patch_size != 0:
            raise ValueError("MAE expects square images divisible by the patch size.")
        grid = height // patch_size
        patches = images.reshape(
            batch_size,
            channels,
            grid,
            patch_size,
            grid,
            patch_size,
        )
        patches = torch.einsum("nchpwq->nhwpqc", patches)
        return patches.reshape(batch_size, grid * grid, patch_size * patch_size * channels)

    def random_masking(
        self,
        tokens: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, dim = tokens.shape
        keep_tokens = int(round(num_tokens * (1.0 - mask_ratio)))
        keep_tokens = max(1, keep_tokens)

        noise = torch.rand(batch_size, num_tokens, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep_tokens]
        kept = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        mask = torch.ones(batch_size, num_tokens, device=tokens.device)
        mask[:, :keep_tokens] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return kept, mask, ids_restore

    def forward_encoder(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.encoder.patch_embed(images)
        tokens = tokens + self.encoder.pos_embed[:, 1:, :]
        tokens, mask, ids_restore = self.random_masking(tokens, self.mask_ratio)

        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.encoder.pos_drop(tokens)
        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)
        return tokens, mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_embed(latent)
        mask_tokens = self.mask_token.repeat(
            decoded.shape[0],
            ids_restore.shape[1] + 1 - decoded.shape[1],
            1,
        )
        decoded_without_cls = torch.cat([decoded[:, 1:, :], mask_tokens], dim=1)
        decoded_without_cls = torch.gather(
            decoded_without_cls,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, decoded.shape[2]),
        )
        decoded = torch.cat([decoded[:, :1, :], decoded_without_cls], dim=1)
        decoded = decoded + self.decoder_pos_embed
        for block in self.decoder_blocks:
            decoded = block(decoded)
        decoded = self.decoder_norm(decoded)
        decoded = self.decoder_pred(decoded)
        return decoded[:, 1:, :]

    def forward_loss(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        targets = self.patchify(images)
        if self.norm_pix_loss:
            mean = targets.mean(dim=-1, keepdim=True)
            variance = targets.var(dim=-1, keepdim=True)
            targets = (targets - mean) / torch.sqrt(variance + 1e-6)
        loss = (predictions - targets) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        latent, mask, ids_restore = self.forward_encoder(images)
        predictions = self.forward_decoder(latent, ids_restore)
        return self.forward_loss(images, predictions, mask)


def parse_csv_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def add_torch_args(parser) -> None:
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODELS,
        help="Comma-separated backbones: resnet18,convnext_tiny,vit_small_patch16_224.",
    )
    parser.add_argument(
        "--vit-init-modes",
        type=str,
        default=DEFAULT_VIT_INITS,
        help="Comma-separated ViT init modes: random,imagenet,mae.",
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
        help="Early-stopping patience measured on validation performance.",
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
        default=DEFAULT_MAE_CHECKPOINT,
        help="Where to save or load the TacVerse MAE ViT checkpoint.",
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


def build_training_config(args) -> TrainingConfig:
    mae_checkpoint = args.mae_checkpoint
    if mae_checkpoint == DEFAULT_MAE_CHECKPOINT:
        mae_checkpoint = default_mae_checkpoint_path(
            image_size=args.image_size,
            mask_ratio=args.mae_mask_ratio,
            mae_epochs=args.mae_epochs,
            mae_max_images=args.mae_max_images,
        )
    if not mae_checkpoint.is_absolute():
        mae_checkpoint = (PROJECT_ROOT / mae_checkpoint).resolve()

    return TrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        allow_random_fallback=args.allow_random_fallback,
        resnet_imagenet_weights=args.resnet_imagenet_weights,
        convnext_imagenet_weights=args.convnext_imagenet_weights,
        vit_imagenet_weights=args.vit_imagenet_weights,
        mae_checkpoint=mae_checkpoint,
        mae_epochs=args.mae_epochs,
        mae_batch_size=args.mae_batch_size,
        mae_lr=args.mae_lr,
        mae_weight_decay=args.mae_weight_decay,
        mae_mask_ratio=args.mae_mask_ratio,
        mae_max_images=args.mae_max_images,
    )


def build_experiment_configs(models_raw: str, vit_init_modes_raw: str) -> list[ExperimentConfig]:
    model_names = parse_csv_list(models_raw)
    vit_init_modes = parse_csv_list(vit_init_modes_raw)

    configs: list[ExperimentConfig] = []
    for model_name in model_names:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        if model_name == VIT_MODEL_NAME:
            for init_mode in vit_init_modes:
                if init_mode not in {"random", "imagenet", "mae"}:
                    raise ValueError(f"Unsupported ViT init mode: {init_mode}")
                configs.append(ExperimentConfig(model_name=model_name, init_mode=init_mode))
        else:
            configs.append(ExperimentConfig(model_name=model_name, init_mode="imagenet"))
    return configs


def experiment_label(model_name: str, init_mode: str) -> str:
    return f"{model_name}_{init_mode}".replace("/", "_").replace("-", "_")


def parse_seed_list(raw_value: str, default_seed: int) -> list[int]:
    if raw_value.strip():
        return [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    return [default_seed]


def with_seed(config: TrainingConfig, seed: int) -> TrainingConfig:
    return replace(config, seed=seed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        raise RuntimeError(
            "CPU training is disabled. Use --device mps, --device cuda, or --device auto on a machine with an accelerator."
        )
    if device.startswith("cuda:"):
        gpu_id = int(device.split(":")[1])
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but CUDA is not available.")
        if gpu_id >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {device}, but only {torch.cuda.device_count()} GPUs are visible.")
        return torch.device(device)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        return torch.device("mps")
    if device != "auto":
        raise RuntimeError(f"Unsupported device '{device}'. Use auto, cuda, cuda:N, or mps.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError(
        "No accelerator is available. CPU training is disabled. Run on a machine with CUDA or MPS, or pass --metadata-only for parsing only."
    )


def load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "encoder_state_dict"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _load_local_pretrained(model: nn.Module, weight_path: Path) -> None:
    state_dict = load_state_dict_file(weight_path)
    model.load_state_dict(state_dict, strict=False)


def _maybe_random_fallback(
    requested_init: str,
    allow_random_fallback: bool,
    error: Exception,
) -> str:
    if requested_init != "imagenet" or not allow_random_fallback:
        raise error
    print(f"ImageNet weights unavailable; falling back to random init. Reason: {error}")
    return "random_fallback"


def collect_tacverse_image_paths(train_only: bool = False) -> list[str]:
    if not train_only:
        image_paths: list[str] = []
        for folder_name in ("Force_Regression", "Shape_Classification", "Grating_Classification"):
            data_dir = PROJECT_ROOT / folder_name
            if not data_dir.exists():
                continue
            image_paths.extend(str(path) for path in sorted(data_dir.rglob("*.jpg")))
        return image_paths

    from common import assign_ordered_split, extract_numeric_suffix

    all_train_paths: list[str] = []

    # Force_Regression: split by sensor using sample_id order
    force_dir = PROJECT_ROOT / "Force_Regression"
    if force_dir.exists():
        rows: list[dict] = []
        for sensor_dir in sorted(force_dir.iterdir()):
            if not sensor_dir.is_dir():
                continue
            csv_path = sensor_dir / f"{sensor_dir.name}.csv"
            if not csv_path.exists():
                continue
            csv_data = pd.read_csv(csv_path)
            csv_data.columns = [c.strip() for c in csv_data.columns]
            for record in csv_data.itertuples(index=False):
                image_name = str(record.image_name).strip()
                image_path = sensor_dir / image_name
                if not image_path.exists():
                    continue
                sample_id = extract_numeric_suffix(Path(image_name).stem)
                rows.append({"sensor": sensor_dir.name, "image_path": str(image_path), "sample_id": sample_id})
        if rows:
            meta = pd.DataFrame(rows).sort_values(["sensor", "sample_id"]).reset_index(drop=True)
            meta = assign_ordered_split(meta, group_cols=["sensor"], order_col="sample_id")
            all_train_paths.extend(meta.loc[meta["split"] == "train", "image_path"].tolist())

    # Shape_Classification: split by (sensor, label) using sample_id order
    shape_dir = PROJECT_ROOT / "Shape_Classification"
    if shape_dir.exists():
        rows = []
        for sensor_dir in sorted(shape_dir.iterdir()):
            if not sensor_dir.is_dir():
                continue
            for class_dir in sorted(sensor_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                for img_path in sorted(class_dir.glob("*.jpg")):
                    sample_id = extract_numeric_suffix(img_path.stem)
                    rows.append({"sensor": sensor_dir.name, "label": class_dir.name,
                                 "image_path": str(img_path), "sample_id": sample_id})
        if rows:
            meta = pd.DataFrame(rows).sort_values(["sensor", "label", "sample_id"]).reset_index(drop=True)
            meta = assign_ordered_split(meta, group_cols=["sensor", "label"], order_col="sample_id")
            all_train_paths.extend(meta.loc[meta["split"] == "train", "image_path"].tolist())

    # Grating_Classification: train = group_id 1-15
    grating_dir = PROJECT_ROOT / "Grating_Classification"
    if grating_dir.exists():
        for sensor_dir in sorted(grating_dir.iterdir()):
            if not sensor_dir.is_dir():
                continue
            for pattern_dir in sorted(sensor_dir.iterdir()):
                if not pattern_dir.is_dir():
                    continue
                for img_path in sorted(pattern_dir.glob("*.jpg")):
                    parts = img_path.stem.split("_")
                    if len(parts) >= 2:
                        try:
                            group_id = int(parts[1])
                            if group_id <= 15:
                                all_train_paths.append(str(img_path))
                        except ValueError:
                            pass

    return sorted(set(all_train_paths))


def mae_signature(config: TrainingConfig) -> dict[str, object]:
    return {
        "pretrain_method": "mae",
        "model_name": VIT_MODEL_NAME,
        "image_size": config.image_size,
        "epochs": config.mae_epochs,
        "batch_size": config.mae_batch_size,
        "lr": config.mae_lr,
        "weight_decay": config.mae_weight_decay,
        "mask_ratio": config.mae_mask_ratio,
        "max_images": config.mae_max_images,
        "input_normalized": True,
    }


def validate_mae_checkpoint(checkpoint_path: Path, config: TrainingConfig) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = mae_signature(config)
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, found {actual_value}")
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise RuntimeError(
            f"Existing MAE checkpoint does not match the requested configuration at {checkpoint_path}: "
            f"{mismatch_text}. Use a different --mae-checkpoint path or remove the stale file."
        )


# ---------------------------------------------------------------------------
# Training utilities: parameter groups, LR schedule, gradient clipping
# ---------------------------------------------------------------------------


def _param_groups_weight_decay(
    model: nn.Module,
    weight_decay: float,
    lr: float,
) -> list[dict]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
    ]


def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(min_lr_ratio, current_step / max(1, warmup_steps))
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------


def ensure_mae_checkpoint(config: TrainingConfig) -> Path:
    checkpoint_path = config.mae_checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        validate_mae_checkpoint(checkpoint_path, config)
        return checkpoint_path

    set_seed(config.seed)

    image_paths = collect_tacverse_image_paths(train_only=True)
    if not image_paths:
        raise RuntimeError("No TacVerse images were found for MAE pretraining.")
    if config.mae_max_images > 0:
        image_paths = image_paths[: config.mae_max_images]

    dataset = ImageOnlyDataset(image_paths, image_size=config.image_size)
    device = resolve_device(config.device)
    loader = DataLoader(
        dataset,
        batch_size=config.mae_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    model = MaskedAutoencoderViT(
        image_size=config.image_size,
        mask_ratio=config.mae_mask_ratio,
    ).to(device)

    param_groups = _param_groups_weight_decay(model, config.mae_weight_decay, config.mae_lr)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    total_steps = config.mae_epochs * len(loader)
    warmup_steps = min(len(loader), total_steps // 5)
    scheduler = _cosine_schedule_with_warmup(optimizer, total_steps, warmup_steps)

    print(
        f"Pretraining TacVerse MAE on {len(dataset)} images for {config.mae_epochs} epochs: {checkpoint_path}"
    )
    model.train()
    for epoch in range(1, config.mae_epochs + 1):
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

        mean_loss = running_loss / max(1, total_samples)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"MAE epoch {epoch}/{config.mae_epochs}: loss={mean_loss:.6f}  lr={current_lr:.2e}")

    checkpoint = {
        "encoder_state_dict": model.encoder.state_dict(),
        "num_images": len(dataset),
        **mae_signature(config),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _create_resnet18(
    num_outputs: int,
    requested_init: str,
    config: TrainingConfig,
) -> tuple[nn.Module, str]:
    resolved_init = requested_init
    if requested_init == "imagenet":
        try:
            if config.resnet_imagenet_weights is not None:
                model = models.resnet18(weights=None)
                _load_local_pretrained(model, config.resnet_imagenet_weights)
            else:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as error:
            resolved_init = _maybe_random_fallback(requested_init, config.allow_random_fallback, error)
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model, resolved_init


def _create_convnext_tiny(
    num_outputs: int,
    requested_init: str,
    config: TrainingConfig,
) -> tuple[nn.Module, str]:
    resolved_init = requested_init
    if requested_init == "imagenet":
        try:
            if config.convnext_imagenet_weights is not None:
                model = models.convnext_tiny(weights=None)
                _load_local_pretrained(model, config.convnext_imagenet_weights)
            else:
                model = models.convnext_tiny(
                    weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
                )
        except Exception as error:
            resolved_init = _maybe_random_fallback(requested_init, config.allow_random_fallback, error)
            model = models.convnext_tiny(weights=None)
    else:
        model = models.convnext_tiny(weights=None)

    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_outputs)
    return model, resolved_init


def _create_vit(
    num_outputs: int,
    requested_init: str,
    config: TrainingConfig,
) -> tuple[nn.Module, str]:
    resolved_init = requested_init
    common_kwargs = {
        "img_size": config.image_size,
        "num_classes": num_outputs,
    }

    if requested_init == "random":
        return timm.create_model(VIT_MODEL_NAME, pretrained=False, **common_kwargs), resolved_init

    if requested_init == "imagenet":
        try:
            if config.vit_imagenet_weights is not None:
                model = timm.create_model(VIT_MODEL_NAME, pretrained=False, **common_kwargs)
                model.load_state_dict(load_state_dict_file(config.vit_imagenet_weights), strict=False)
                return model, resolved_init
            return timm.create_model(VIT_MODEL_NAME, pretrained=True, **common_kwargs), resolved_init
        except Exception as error:
            resolved_init = _maybe_random_fallback(requested_init, config.allow_random_fallback, error)
            return timm.create_model(VIT_MODEL_NAME, pretrained=False, **common_kwargs), resolved_init

    if requested_init == "mae":
        checkpoint_path = ensure_mae_checkpoint(config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        encoder_state = checkpoint.get("encoder_state_dict", checkpoint)
        model = timm.create_model(VIT_MODEL_NAME, pretrained=False, **common_kwargs)
        model.load_state_dict(encoder_state, strict=False)
        return model, resolved_init

    raise ValueError(f"Unsupported ViT init mode: {requested_init}")


def create_model(
    model_name: str,
    num_outputs: int,
    requested_init: str,
    config: TrainingConfig,
) -> tuple[nn.Module, str]:
    if model_name == "resnet18":
        return _create_resnet18(num_outputs, requested_init, config)
    if model_name == "convnext_tiny":
        return _create_convnext_tiny(num_outputs, requested_init, config)
    if model_name == VIT_MODEL_NAME:
        return _create_vit(num_outputs, requested_init, config)
    raise ValueError(f"Unsupported model: {model_name}")


def build_label_mapping(labels: Sequence) -> tuple[list[str], dict[str, int]]:
    class_names = [str(label) for label in sorted(pd.Series(labels).astype(str).unique())]
    label_to_index = {label: index for index, label in enumerate(class_names)}
    return class_names, label_to_index


def make_loader(
    frame: pd.DataFrame,
    image_size: int,
    task_type: str,
    label_col: str | None,
    target_cols: Sequence[str] | None,
    label_to_index: dict[str, int] | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    normalize: bool = True,
    is_training: bool = False,
) -> DataLoader:
    dataset = TacverseDataset(
        frame=frame,
        image_size=image_size,
        task_type=task_type,
        label_col=label_col,
        target_cols=target_cols,
        label_to_index=label_to_index,
        normalize=normalize,
        is_training=is_training,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _classification_class_weights(train_frame: pd.DataFrame, label_col: str, label_to_index: dict[str, int]):
    counts = np.zeros(len(label_to_index), dtype=np.float32)
    for label, count in train_frame[label_col].astype(str).value_counts().items():
        counts[label_to_index[label]] = float(count)
    nonzero = counts > 0
    weights = np.zeros_like(counts)
    weights[nonzero] = counts[nonzero].sum() / (len(counts[nonzero]) * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_grad_norm: float = 0.0,
    epoch_info: str = "",
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    num_batches = len(loader)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = inputs.size(0)
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    avg_loss = running_loss / max(1, total_samples)
    if epoch_info:
        print(f"    {epoch_info} loss={avg_loss:.4f}", flush=True)
    return avg_loss


def _predict_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true_labels: list[np.ndarray] = []
    pred_labels: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            predictions = logits.argmax(dim=1).cpu().numpy()
            true_labels.append(targets.cpu().numpy())
            pred_labels.append(predictions)
    return np.concatenate(true_labels), np.concatenate(pred_labels)


def _predict_regression(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true_values: list[np.ndarray] = []
    pred_values: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            true_values.append(targets.cpu().numpy())
            pred_values.append(outputs)
    return np.vstack(true_values), np.vstack(pred_values)


def train_classification_experiment(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_col: str,
    model_name: str,
    init_mode: str,
    config: TrainingConfig,
    class_weight_mode: str | None = None,
) -> dict:
    if train_frame.empty or test_frame.empty:
        raise ValueError("train_frame and test_frame must be non-empty.")

    set_seed(config.seed)
    device = resolve_device(config.device)

    all_paths = (
        train_frame["image_path"].tolist()
        + val_frame["image_path"].tolist()
        + test_frame["image_path"].tolist()
    )
    prefetch_images(all_paths)

    label_names, label_to_index = build_label_mapping(
        pd.concat([train_frame[label_col], val_frame[label_col], test_frame[label_col]], ignore_index=True)
    )
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_frame,
        config.image_size,
        "classification",
        label_col,
        None,
        label_to_index,
        config.batch_size,
        True,
        config.num_workers,
        pin_memory,
        is_training=True,
    )
    val_loader = None
    if not val_frame.empty:
        val_loader = make_loader(
            val_frame,
            config.image_size,
            "classification",
            label_col,
            None,
            label_to_index,
            config.batch_size,
            False,
            config.num_workers,
            pin_memory,
        )
    test_loader = make_loader(
        test_frame,
        config.image_size,
        "classification",
        label_col,
        None,
        label_to_index,
        config.batch_size,
        False,
        config.num_workers,
        pin_memory,
    )

    model, resolved_init_mode = create_model(
        model_name=model_name,
        num_outputs=len(label_names),
        requested_init=init_mode,
        config=config,
    )
    model = model.to(device)

    weights = None
    if class_weight_mode == "balanced":
        weights = _classification_class_weights(train_frame, label_col, label_to_index).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    param_groups = _param_groups_weight_decay(model, config.weight_decay, config.lr)
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = config.epochs * len(train_loader)
    warmup_steps = min(len(train_loader), total_steps // 5)
    scheduler = _cosine_schedule_with_warmup(optimizer, total_steps, warmup_steps)

    best_state = copy.deepcopy(model.state_dict())
    best_val_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        _train_one_epoch(model, train_loader, optimizer, loss_fn, device,
                         scheduler=scheduler, max_grad_norm=1.0,
                         epoch_info=f"epoch {epoch}/{config.epochs}")

        if val_loader is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            continue

        y_true_val, y_pred_val = _predict_classification(model, val_loader, device)
        val_accuracy = float(np.mean(y_true_val == y_pred_val))

        if val_accuracy > best_val_score:
            best_val_score = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    y_true_test, y_pred_test = _predict_classification(model, test_loader, device)
    return {
        "y_true": y_true_test,
        "y_pred": y_pred_test,
        "class_names": label_names,
        "resolved_init_mode": resolved_init_mode,
        "device": device.type,
        "best_epoch": best_epoch,
    }


def train_regression_experiment(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_cols: Sequence[str],
    model_name: str,
    init_mode: str,
    config: TrainingConfig,
) -> dict:
    if train_frame.empty or test_frame.empty:
        raise ValueError("train_frame and test_frame must be non-empty.")

    set_seed(config.seed)
    device = resolve_device(config.device)

    all_paths = (
        train_frame["image_path"].tolist()
        + val_frame["image_path"].tolist()
        + test_frame["image_path"].tolist()
    )
    prefetch_images(all_paths)

    normalizer = RegressionTargetNormalizer.fit(train_frame, target_cols)
    normalized_train_frame = normalizer.transform_frame(train_frame, target_cols)
    normalized_val_frame = normalizer.transform_frame(val_frame, target_cols)
    normalized_test_frame = normalizer.transform_frame(test_frame, target_cols)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        normalized_train_frame,
        config.image_size,
        "regression",
        None,
        target_cols,
        None,
        config.batch_size,
        True,
        config.num_workers,
        pin_memory,
        is_training=True,
    )
    val_loader = None
    if not normalized_val_frame.empty:
        val_loader = make_loader(
            normalized_val_frame,
            config.image_size,
            "regression",
            None,
            target_cols,
            None,
            config.batch_size,
            False,
            config.num_workers,
            pin_memory,
        )
    test_loader = make_loader(
        normalized_test_frame,
        config.image_size,
        "regression",
        None,
        target_cols,
        None,
        config.batch_size,
        False,
        config.num_workers,
        pin_memory,
    )

    model, resolved_init_mode = create_model(
        model_name=model_name,
        num_outputs=len(tuple(target_cols)),
        requested_init=init_mode,
        config=config,
    )
    model = model.to(device)
    loss_fn = nn.MSELoss()

    param_groups = _param_groups_weight_decay(model, config.weight_decay, config.lr)
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = config.epochs * len(train_loader)
    warmup_steps = min(len(train_loader), total_steps // 5)
    scheduler = _cosine_schedule_with_warmup(optimizer, total_steps, warmup_steps)

    best_state = copy.deepcopy(model.state_dict())
    best_val_score = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        _train_one_epoch(model, train_loader, optimizer, loss_fn, device,
                         scheduler=scheduler, max_grad_norm=1.0,
                         epoch_info=f"epoch {epoch}/{config.epochs}")

        if val_loader is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            continue

        y_true_val, y_pred_val = _predict_regression(model, val_loader, device)
        y_true_val = normalizer.inverse_transform_array(y_true_val)
        y_pred_val = normalizer.inverse_transform_array(y_pred_val)
        val_rmse = float(np.sqrt(np.mean((y_pred_val - y_true_val) ** 2)))

        if val_rmse < best_val_score:
            best_val_score = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    y_true_test, y_pred_test = _predict_regression(model, test_loader, device)
    y_true_test = normalizer.inverse_transform_array(y_true_test)
    y_pred_test = normalizer.inverse_transform_array(y_pred_test)
    return {
        "y_true": y_true_test,
        "y_pred": y_pred_test,
        "resolved_init_mode": resolved_init_mode,
        "device": device.type,
        "best_epoch": best_epoch,
    }
