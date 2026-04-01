from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    PROJECT_ROOT,
    classification_metrics,
    ensure_results_dir,
    format_sensor_list,
    select_fewshot_by_label,
    size_mae,
)
from torch_common import (
    add_torch_args,
    build_experiment_configs,
    build_training_config,
    ensure_mae_checkpoint,
    experiment_label,
    parse_seed_list,
    train_classification_experiment,
    with_seed,
)


GRATING_DIR = PROJECT_ROOT / "Grating_Classification"
SHARED_SENSORS = ["GelsightMarker", "GelsightNoMarker", "MagicGripper", "ViTacTip"]
SHARED_RAW_PATTERNS = {
    "Blackdot",
    "Blackline",
    "Blackdot_nomarker",
    "Blackline_nomarker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grating classification benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "grating_classification_results.csv",
        help="Where to save the summary CSV.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize images to a square of this size.",
    )
    parser.add_argument(
        "--sensors",
        type=str,
        default="",
        help="Optional comma-separated sensor subset.",
    )
    parser.add_argument(
        "--protocols",
        type=str,
        default="within,cross,loso,fewshot",
        help="Comma-separated protocols: within,cross,loso,fewshot.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="pattern,size,joint",
        help="Comma-separated variants: pattern,size,joint.",
    )
    parser.add_argument(
        "--fewshot-shots",
        type=str,
        default="1,5,10",
        help="Comma-separated support groups per class for few-shot adaptation.",
    )
    parser.add_argument(
        "--target-sensors",
        type=str,
        default="",
        help="Optional comma-separated target sensor subset. When set, only these sensors are used as evaluation targets; all sensors remain available as sources.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only parse and print metadata statistics.",
    )
    add_torch_args(parser)
    return parser.parse_args()


def parse_filename(image_path: Path) -> tuple[str, int, int]:
    parts = image_path.stem.split("_")
    if len(parts) == 3:
        size_str, group_str, frame_str = parts
        return size_str, int(group_str), int(frame_str)
    if len(parts) == 2:
        size_str, group_str = parts
        return size_str, int(group_str), 0
    raise ValueError(f"Unexpected grating filename format: {image_path.name}")


def infer_pattern_family(pattern_raw: str) -> str:
    return "dot" if "dot" in pattern_raw.lower() else "line"


def infer_appearance(pattern_raw: str) -> str:
    pattern_lower = pattern_raw.lower()
    if "nomarker" in pattern_lower:
        return "nomarker"
    if pattern_raw.startswith("Tran"):
        return "transparent"
    return "black"


def build_grating_metadata() -> pd.DataFrame:
    rows: list[dict] = []

    for sensor_dir in sorted(GRATING_DIR.iterdir()):
        if not sensor_dir.is_dir():
            continue
        for pattern_dir in sorted(sensor_dir.iterdir()):
            if not pattern_dir.is_dir():
                continue
            for image_path in sorted(pattern_dir.glob("*.jpg")):
                size_str, group_id, frame_id = parse_filename(image_path)
                rows.append(
                    {
                        "sensor": sensor_dir.name,
                        "pattern_raw": pattern_dir.name,
                        "pattern_family": infer_pattern_family(pattern_dir.name),
                        "appearance": infer_appearance(pattern_dir.name),
                        "size_str": size_str,
                        "size_value": float(size_str),
                        "group_id": group_id,
                        "frame_id": frame_id,
                        "image_path": str(image_path),
                        "image_name": image_path.name,
                    }
                )

    metadata = pd.DataFrame(rows).sort_values(
        ["sensor", "pattern_raw", "size_value", "group_id", "frame_id"]
    )
    return metadata.reset_index(drop=True)


def build_shared_benchmark(metadata: pd.DataFrame) -> pd.DataFrame:
    subset = metadata[
        metadata["sensor"].isin(SHARED_SENSORS)
        & metadata["pattern_raw"].isin(SHARED_RAW_PATTERNS)
        & metadata["group_id"].between(1, 25)
        & metadata["frame_id"].between(1, 4)
    ].copy()

    complete_groups = (
        subset.groupby(["sensor", "pattern_raw", "size_str", "group_id"])
        .size()
        .reset_index(name="num_frames")
    )
    complete_groups = complete_groups[complete_groups["num_frames"] == 4].drop(
        columns="num_frames"
    )
    subset = subset.merge(
        complete_groups,
        on=["sensor", "pattern_raw", "size_str", "group_id"],
        how="inner",
    )

    size_sets = [
        set(group["size_str"].unique())
        for _, group in subset.groupby(["sensor", "pattern_family"], sort=False)
    ]
    if not size_sets:
        return subset.iloc[:0].copy()
    common_sizes = set.intersection(*size_sets)
    subset = subset[subset["size_str"].isin(common_sizes)].copy()

    subset["split"] = np.where(
        subset["group_id"] <= 15,
        "train",
        np.where(subset["group_id"] <= 20, "val", "test"),
    )
    subset["support_id"] = (
        subset["sensor"]
        + "|"
        + subset["pattern_raw"]
        + "|"
        + subset["size_str"]
        + "|"
        + subset["group_id"].astype(str)
    )
    subset["size_label"] = subset["size_str"]
    subset["joint_label"] = subset["pattern_family"] + "|" + subset["size_str"]
    return subset.reset_index(drop=True)


def parse_int_list(raw_value: str) -> list[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def fit_and_score(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_col: str,
    model_name: str,
    init_mode: str,
    training_config,
) -> dict:
    outputs = train_classification_experiment(
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        label_col=label_col,
        model_name=model_name,
        init_mode=init_mode,
        config=training_config,
    )
    class_names = outputs["class_names"]
    true_labels = [class_names[index] for index in outputs["y_true"]]
    pred_labels = [class_names[index] for index in outputs["y_pred"]]
    metrics = classification_metrics(true_labels, pred_labels)

    if label_col == "size_label":
        metrics["size_mae"] = size_mae(
            [float(label) for label in true_labels],
            [float(label) for label in pred_labels],
        )
        metrics["pattern_accuracy_from_joint"] = np.nan
    elif label_col == "joint_label":
        metrics["size_mae"] = size_mae(
            [float(label.split("|", 1)[1]) for label in true_labels],
            [float(label.split("|", 1)[1]) for label in pred_labels],
        )
        metrics["pattern_accuracy_from_joint"] = float(
            np.mean(
                np.asarray([label.split("|", 1)[0] for label in true_labels])
                == np.asarray([label.split("|", 1)[0] for label in pred_labels])
            )
        )
    else:
        metrics["size_mae"] = np.nan
        metrics["pattern_accuracy_from_joint"] = np.nan

    return {
        **metrics,
        "init_mode": outputs["resolved_init_mode"],
        "device": outputs["device"],
        "best_epoch": outputs["best_epoch"],
    }


def append_row(
    rows: list[dict],
    variant: str,
    training_config,
    protocol: str,
    source_sensors: str,
    target_sensor: str,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_col: str,
    model_name: str,
    init_mode_requested: str,
    fewshot_value,
    fewshot_unit: str,
    support_units: int,
    support_samples: int,
    num_classes: int,
) -> None:
    metrics = fit_and_score(
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        label_col=label_col,
        model_name=model_name,
        init_mode=init_mode_requested,
        training_config=training_config,
    )
    rows.append(
        {
            "task": "Grating_Classification",
            "seed": training_config.seed,
            "variant": variant,
            "model_name": model_name,
            "model_label": experiment_label(model_name, metrics["init_mode"]),
            "init_mode_requested": init_mode_requested,
            "init_mode": metrics.pop("init_mode"),
            "device": metrics.pop("device"),
            "best_epoch": metrics.pop("best_epoch"),
            "protocol": protocol,
            "source_sensors": source_sensors,
            "target_sensor": target_sensor,
            "fewshot_value": fewshot_value,
            "fewshot_unit": fewshot_unit,
            "support_units": support_units,
            "support_samples": support_samples,
            "num_classes": num_classes,
            "train_samples": int(len(train_frame)),
            "val_samples": int(len(val_frame)),
            "test_samples": int(len(test_frame)),
            **metrics,
        }
    )
    row = rows[-1]
    print(
        f"  [{len(rows):>4d}] {protocol:<22s} {source_sensors:<30s} → {target_sensor:<20s} "
        f"acc={row.get('accuracy', 0):.4f}  f1={row.get('macro_f1', 0):.4f}  "
        f"epoch={row['best_epoch']}"
    )


def run_variant(
    metadata: pd.DataFrame,
    sensors: list[str],
    protocols: set[str],
    variant: str,
    label_col: str,
    fewshot_shots: list[int],
    seeds: list[int],
    experiment_configs,
    training_config,
    output_path: Path | None = None,
    target_sensors: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    subset = metadata[metadata["sensor"].isin(sensors)].reset_index(drop=True)
    num_classes = int(subset[label_col].nunique())
    eval_targets = target_sensors if target_sensors else sensors

    if any(config.init_mode == "mae" for config in experiment_configs):
        ensure_mae_checkpoint(training_config)

    for seed in seeds:
        seeded_training_config = with_seed(training_config, seed)
        for config in experiment_configs:
            print(f"Running grating {variant} for {config.model_name} / {config.init_mode} / seed {seed}")

            if "within" in protocols:
                for sensor in eval_targets:
                    train_frame = subset[
                        subset["sensor"].eq(sensor) & subset["split"].eq("train")
                    ].reset_index(drop=True)
                    val_frame = subset[
                        subset["sensor"].eq(sensor) & subset["split"].eq("val")
                    ].reset_index(drop=True)
                    test_frame = subset[
                        subset["sensor"].eq(sensor) & subset["split"].eq("test")
                    ].reset_index(drop=True)
                    if train_frame.empty or test_frame.empty:
                        continue
                    append_row(
                        rows=rows,
                        variant=variant,
                        training_config=seeded_training_config,
                        protocol="within_sensor",
                        source_sensors=sensor,
                        target_sensor=sensor,
                        train_frame=train_frame,
                        val_frame=val_frame,
                        test_frame=test_frame,
                        label_col=label_col,
                        model_name=config.model_name,
                        init_mode_requested=config.init_mode,
                        fewshot_value=np.nan,
                        fewshot_unit="",
                        support_units=0,
                        support_samples=0,
                        num_classes=num_classes,
                    )

            if "cross" in protocols:
                for source_sensor in sensors:
                    for target_sensor in eval_targets:
                        if source_sensor == target_sensor:
                            continue
                        train_frame = subset[
                            subset["sensor"].eq(source_sensor) & subset["split"].eq("train")
                        ].reset_index(drop=True)
                        val_frame = subset[
                            subset["sensor"].eq(source_sensor) & subset["split"].eq("val")
                        ].reset_index(drop=True)
                        test_frame = subset[
                            subset["sensor"].eq(target_sensor) & subset["split"].eq("test")
                        ].reset_index(drop=True)
                        if train_frame.empty or test_frame.empty:
                            continue
                        append_row(
                            rows=rows,
                            variant=variant,
                            training_config=seeded_training_config,
                            protocol="cross_sensor",
                            source_sensors=source_sensor,
                            target_sensor=target_sensor,
                            train_frame=train_frame,
                            val_frame=val_frame,
                            test_frame=test_frame,
                            label_col=label_col,
                            model_name=config.model_name,
                            init_mode_requested=config.init_mode,
                            fewshot_value=np.nan,
                            fewshot_unit="",
                            support_units=0,
                            support_samples=0,
                            num_classes=num_classes,
                        )

            if "loso" in protocols:
                for target_sensor in eval_targets:
                    source_sensors = [sensor for sensor in sensors if sensor != target_sensor]
                    train_frame = subset[
                        subset["sensor"].isin(source_sensors) & subset["split"].eq("train")
                    ].reset_index(drop=True)
                    val_frame = subset[
                        subset["sensor"].isin(source_sensors) & subset["split"].eq("val")
                    ].reset_index(drop=True)
                    test_frame = subset[
                        subset["sensor"].eq(target_sensor) & subset["split"].eq("test")
                    ].reset_index(drop=True)
                    if train_frame.empty or test_frame.empty:
                        continue
                    append_row(
                        rows=rows,
                        variant=variant,
                        training_config=seeded_training_config,
                        protocol="leave_one_sensor_out",
                        source_sensors=format_sensor_list(source_sensors),
                        target_sensor=target_sensor,
                        train_frame=train_frame,
                        val_frame=val_frame,
                        test_frame=test_frame,
                        label_col=label_col,
                        model_name=config.model_name,
                        init_mode_requested=config.init_mode,
                        fewshot_value=np.nan,
                        fewshot_unit="",
                        support_units=0,
                        support_samples=0,
                        num_classes=num_classes,
                    )

            if "fewshot" in protocols:
                for target_sensor in eval_targets:
                    source_sensors = [sensor for sensor in sensors if sensor != target_sensor]
                    source_train = subset[
                        subset["sensor"].isin(source_sensors) & subset["split"].eq("train")
                    ]
                    source_val = subset[
                        subset["sensor"].isin(source_sensors) & subset["split"].eq("val")
                    ]
                    target_support_pool = subset[
                        subset["sensor"].eq(target_sensor) & subset["split"].eq("train")
                    ]
                    test_frame = subset[
                        subset["sensor"].eq(target_sensor) & subset["split"].eq("test")
                    ].reset_index(drop=True)

                    if source_train.empty or target_support_pool.empty or test_frame.empty:
                        continue

                    for shots in fewshot_shots:
                        eligible_mask = pd.Series(False, index=subset.index)
                        eligible_mask.loc[target_support_pool.index] = True
                        support_mask = select_fewshot_by_label(
                            metadata=subset,
                            eligible_mask=eligible_mask,
                            label_col=label_col,
                            num_units_per_label=shots,
                            unit_col="support_id",
                            sort_cols=["size_value", "group_id", "frame_id"],
                            seed=seed,
                        )
                        support_frame = subset[support_mask].reset_index(drop=True)
                        train_frame = pd.concat([source_train, support_frame], ignore_index=True)
                        append_row(
                            rows=rows,
                            variant=variant,
                            training_config=seeded_training_config,
                            protocol="fewshot_adaptation",
                            source_sensors=format_sensor_list(source_sensors),
                            target_sensor=target_sensor,
                            train_frame=train_frame,
                            val_frame=source_val.reset_index(drop=True),
                            test_frame=test_frame,
                            label_col=label_col,
                            model_name=config.model_name,
                            init_mode_requested=config.init_mode,
                            fewshot_value=shots,
                            fewshot_unit="groups_per_class",
                            support_units=int(support_frame["support_id"].nunique()),
                            support_samples=int(len(support_frame)),
                            num_classes=num_classes,
                        )

            if output_path is not None and rows:
                pd.DataFrame(rows).to_csv(output_path, index=False)

    return pd.DataFrame(rows)


def print_metadata_summary(metadata: pd.DataFrame) -> None:
    summary = (
        metadata.groupby(["sensor", "pattern_family"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(summary)
    print("Size levels:", ", ".join(sorted(metadata["size_str"].unique(), key=float)))


def main() -> None:
    args = parse_args()
    metadata = build_shared_benchmark(build_grating_metadata())

    if args.sensors:
        sensor_subset = [sensor.strip() for sensor in args.sensors.split(",") if sensor.strip()]
        metadata = metadata[metadata["sensor"].isin(sensor_subset)].reset_index(drop=True)

    if metadata.empty:
        raise RuntimeError("No grating benchmark samples were found.")

    print_metadata_summary(metadata)
    if args.metadata_only:
        return

    sensors = sorted(metadata["sensor"].unique())
    protocols = {protocol.strip() for protocol in args.protocols.split(",") if protocol.strip()}
    variants = {variant.strip() for variant in args.variants.split(",") if variant.strip()}
    fewshot_shots = parse_int_list(args.fewshot_shots)
    seeds = parse_seed_list(args.seeds, args.seed)
    experiment_configs = build_experiment_configs(args.models, args.vit_init_modes)
    training_config = build_training_config(args)
    target_sensors = [s.strip() for s in args.target_sensors.split(",") if s.strip()] or None

    variant_map = {
        "pattern": "pattern_family",
        "size": "size_label",
        "joint": "joint_label",
    }

    frames: list[pd.DataFrame] = []
    for variant_name, label_col in variant_map.items():
        if variant_name not in variants:
            continue
        frames.append(
            run_variant(
                metadata=metadata,
                sensors=sensors,
                protocols=protocols,
                variant=variant_name,
                label_col=label_col,
                fewshot_shots=fewshot_shots,
                seeds=seeds,
                experiment_configs=experiment_configs,
                training_config=training_config,
                output_path=args.output,
                target_sensors=target_sensors,
            )
        )

    if not frames:
        raise RuntimeError("No grating experiments were produced. Check sensor and variant filters.")

    results = pd.concat(frames, ignore_index=True)
    results.insert(0, "image_size", args.image_size)
    ensure_results_dir()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Saved {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
