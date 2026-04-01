# TacVerse Benchmark Code

This repository contains PyTorch scripts for running tactile image benchmarks on the TacVerse dataset.

## Included scripts

- `force_regression.py`: predict `fx`, `fy`, `fz` from tactile images.
- `shape_classification.py`: shape classification settings.
- `grating_classification.py`: grating `pattern`, `size`, and `joint` classification.
- `mae_pretrain.py`: MAE pretraining for scratch.
- `run_all.py`: run the three downstream benchmarks in sequence.
- `common.py` and `torch_common.py`: shared data loading, metrics, and training utilities.

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- timm
- numpy
- pandas
- Pillow

An accelerator is required for training. The current code supports CUDA and Apple MPS. CPU training is disabled.

## Expected data layout

The scripts treat the parent directory of this folder as `PROJECT_ROOT`. A typical layout is:

Dataset download:
https://huggingface.co/datasets/Lan-2025/Tactile

```text
project_root/
  Force_Regression/
  Grating_Classification/
  Shape_Classification/
  lan_tactile-main/
    README.md
    *.py
```

Outputs are written to `project_root/results/`.

## Quick start

Run a metadata check first:

```bash
python force_regression.py --device cuda --metadata-only
python shape_classification.py --device cuda --metadata-only
python grating_classification.py --device cuda --metadata-only
```

Run the benchmarks:

```bash
python force_regression.py --device cuda
python shape_classification.py --device cuda
python grating_classification.py --device cuda
python run_all.py --device cuda
```

MAE pretraining:

```bash
python mae_pretrain.py --device cuda --image-size 224
```

## Notes

- Supported backbones: `resnet18`, `convnext_tiny`, `vit_small_patch16_224`
- ViT init modes: `random`, `imagenet`, `mae`
- Default protocols are `within`, `cross`, `loso`, and `fewshot`
- If ImageNet weights are unavailable offline, provide local weight files or use `--allow-random-fallback`

## Outputs

The main result files are saved as CSV files under `results/`, for example:

- `force_regression_results.csv`
- `shape_classification_results.csv`
- `grating_classification_results.csv`
