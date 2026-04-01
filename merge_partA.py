"""Reconstruct Part A CSVs purely from Part A logs.

Parses completed experiment results from partA_force.log and partA_shape.log.
Grating Part A was never started (no log).

Outputs what we have and reports what's missing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"

# ── Part A target config (from experiment_plan_minimal.md) ──
PART_A_CONFIG = {
    "force": {
        "target": "MagicGripper",
        "model_name": "vit_small_patch16_224",
        "init_mode": "imagenet",
        "fewshot_values": [0.005, 0.01, 0.05, 0.1],
        "fewshot_unit": "target_train_ratio",
    },
    "grating": {
        "target": "GelsightMarker",
        "model_name": "vit_small_patch16_224",
        "init_mode": "imagenet",
        "variant": "pattern",
        "fewshot_values": [1, 5, 10],
        "fewshot_unit": "groups_per_class",
    },
    "shape": {
        "target": "ViTacTip",
        "model_name": "vit_small_patch16_224",
        "init_mode": "imagenet",
        "variant": "longtail",
        "fewshot_values": [1, 5, 10, 20],
        "fewshot_unit": "shots_per_class",
    },
}

# ── Log parsing ──
LOG_PATTERN = re.compile(
    r"\[\s*(\d+)\]\s+"
    r"(within_sensor|cross_sensor|leave_one_sensor_out|fewshot_adaptation)\s+"
    r"(\S+(?:;\S+)*)\s+→\s+(\S+)\s+"
    r"(?:RMSE=([0-9.]+)\s+MAE=([0-9.]+)|acc=([0-9.]+)\s+f1=([0-9.]+))"
    r"\s+epoch=(\d+)"
)


def parse_force_log() -> pd.DataFrame:
    log_path = LOG_DIR / "partA_force.log"
    if not log_path.exists():
        return pd.DataFrame()
    text = log_path.read_text()
    rows = []
    cfg = PART_A_CONFIG["force"]
    fewshot_idx = 0

    for m in LOG_PATTERN.finditer(text):
        idx, protocol, source, target, rmse, mae, acc, f1, epoch = m.groups()
        row = {
            "image_size": 224,
            "task": "Force_Regression",
            "seed": 42,
            "model_name": cfg["model_name"],
            "model_label": f"{cfg['model_name']}_{cfg['init_mode']}",
            "init_mode_requested": cfg["init_mode"],
            "init_mode": cfg["init_mode"],
            "device": "cuda",
            "best_epoch": int(epoch),
            "protocol": protocol,
            "source_sensors": source,
            "target_sensor": target,
            "fewshot_value": "",
            "fewshot_unit": "",
            "support_units": 0,
            "support_samples": 0,
            "rmse": float(rmse) if rmse else None,
            "mae": float(mae) if mae else None,
        }
        if protocol == "fewshot_adaptation":
            if fewshot_idx < len(cfg["fewshot_values"]):
                row["fewshot_value"] = cfg["fewshot_values"][fewshot_idx]
                row["fewshot_unit"] = cfg["fewshot_unit"]
                fewshot_idx += 1
        rows.append(row)

    return pd.DataFrame(rows)


def parse_shape_log() -> pd.DataFrame:
    log_path = LOG_DIR / "partA_shape.log"
    if not log_path.exists():
        return pd.DataFrame()
    text = log_path.read_text()
    rows = []
    cfg = PART_A_CONFIG["shape"]
    fewshot_idx = 0

    for m in LOG_PATTERN.finditer(text):
        idx, protocol, source, target, rmse, mae, acc, f1, epoch = m.groups()
        row = {
            "image_size": 224,
            "task": "Shape_Classification",
            "seed": 42,
            "variant": cfg["variant"],
            "model_name": cfg["model_name"],
            "model_label": f"{cfg['model_name']}_{cfg['init_mode']}",
            "init_mode_requested": cfg["init_mode"],
            "init_mode": cfg["init_mode"],
            "device": "cuda",
            "best_epoch": int(epoch),
            "protocol": protocol,
            "source_sensors": source,
            "target_sensor": target,
            "fewshot_value": "",
            "fewshot_unit": "",
            "support_units": 0,
            "support_samples": 0,
            "accuracy": float(acc) if acc else None,
            "macro_f1": float(f1) if f1 else None,
        }
        if protocol == "fewshot_adaptation":
            if fewshot_idx < len(cfg["fewshot_values"]):
                row["fewshot_value"] = cfg["fewshot_values"][fewshot_idx]
                row["fewshot_unit"] = cfg["fewshot_unit"]
                fewshot_idx += 1
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Part A — Reconstructing from logs only")
    print("=" * 60)

    # ── Force ──
    print("\n── Force Regression (target=MagicGripper) ──")
    force = parse_force_log()
    if not force.empty:
        by_protocol = force.groupby("protocol").size()
        for p, n in by_protocol.items():
            print(f"  ✓ {p}: {n} row(s)")
        force.to_csv(RESULTS_DIR / "partA_force.csv", index=False)
        print(f"  → Saved {len(force)} rows to partA_force.csv")
    else:
        print("  ✗ No partA_force.log found")

    force_done_fewshot = len(force[force["protocol"] == "fewshot_adaptation"]) if not force.empty else 0
    force_missing = len(PART_A_CONFIG["force"]["fewshot_values"]) - force_done_fewshot
    missing_ratios = PART_A_CONFIG["force"]["fewshot_values"][force_done_fewshot:]

    # ── Shape ──
    print("\n── Shape Classification (target=ViTacTip) ──")
    shape = parse_shape_log()
    if not shape.empty:
        by_protocol = shape.groupby("protocol").size()
        for p, n in by_protocol.items():
            print(f"  ✓ {p}: {n} row(s)")
        shape.to_csv(RESULTS_DIR / "partA_shape.csv", index=False)
        print(f"  → Saved {len(shape)} rows to partA_shape.csv")
    else:
        print("  ✗ No partA_shape.log found")

    shape_done_fewshot = len(shape[shape["protocol"] == "fewshot_adaptation"]) if not shape.empty else 0
    shape_missing = len(PART_A_CONFIG["shape"]["fewshot_values"]) - shape_done_fewshot
    missing_shots = PART_A_CONFIG["shape"]["fewshot_values"][shape_done_fewshot:]

    # ── Grating ──
    print("\n── Grating Classification (target=GelsightMarker) ──")
    print("  ✗ No partA_grating.log — never started")
    print("  → Missing: ALL 8 experiments")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Still needs GPU time:")
    print("=" * 60)
    print(f"  Force fewshot:   {force_missing} experiments  --fewshot-ratios {','.join(str(r) for r in missing_ratios)}")
    print(f"  Shape fewshot:   {shape_missing} experiments  --fewshot-shots {','.join(str(s) for s in missing_shots)}")
    print(f"  Grating:         8 experiments   (ALL — within,cross,loso,fewshot)")
    total = force_missing + shape_missing + 8
    print(f"  Total:           {total} experiments (was 32)")
    print()
    print("Note: log-parsed rows only have rmse/mae (force) or acc/f1 (shape).")
    print("      Missing: pearson_mean, balanced_accuracy, train/val/test_samples, per-axis metrics.")
    print("      If paper tables need those columns, within/cross/loso also need re-running.")


if __name__ == "__main__":
    main()
