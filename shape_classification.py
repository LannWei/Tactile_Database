from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    PROJECT_ROOT,
    assign_ordered_split,
    classification_metrics,
    ensure_results_dir,
    extract_numeric_suffix,
    format_sensor_list,
    select_fewshot_by_label,
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


SHAPE_DIR = PROJECT_ROOT / "Shape_Classification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shape classification benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "shape_classification_results.csv",
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
        default="balanced,longtail",
        help="Comma-separated variants: balanced,longtail.",
    )
    parser.add_argument(
        "--fewshot-shots",
        type=str,
        default="1,5,10,20",
        help="Comma-separated shots-per-class used for few-shot adaptation.",
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


def build_shape_metadata() -> pd.DataFrame:
    rows: list[dict] = []

    for sensor_dir in sorted(SHAPE_DIR.iterdir()):
        if not sensor_dir.is_dir():
            continue
        for class_dir in sorted(sensor_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.glob("*.jpg")):
                sample_id = extract_numeric_suffix(image_path.stem)
                rows.append(
                    {
                        "sensor": sensor_dir.name,
                        "label": class_dir.name,
                        "image_path": str(image_path),
                        "image_name": image_path.name,
                        "sample_id": sample_id,
                    }
                )

    metadata = pd.DataFrame(rows).sort_values(["sensor", "label", "sample_id"]).reset_index(
        drop=True
    )
    return assign_ordered_split(
        metadata,
        group_cols=["sensor", "label"],
        order_col="sample_id",
    )


def infer_balanced_sensors(metadata: pd.DataFrame) -> list[str]:
    counts = metadata.groupby(["sensor", "label"]).size().unstack(fill_value=0)
    balanced_mask = (counts.min(axis=1) > 0) & (counts.nunique(axis=1) == 1)
    return sorted(counts.index[balanced_mask].tolist())


def parse_int_list(raw_value: str) -> list[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def fit_and_score(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    model_name: str,
    init_mode: str,
    training_config,
    class_weight: str | None,
) -> dict[str, float]:
    outputs = train_classification_experiment(
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        label_col="label",
        model_name=model_name,
        init_mode=init_mode,
        config=training_config,
        class_weight_mode=class_weight,
    )
    metrics = classification_metrics(outputs["y_true"], outputs["y_pred"])
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
    model_name: str,
    init_mode_requested: str,
    class_weight: str | None,
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
        model_name=model_name,
        init_mode=init_mode_requested,
        training_config=training_config,
        class_weight=class_weight,
    )
    rows.append(
        {
            "task": "Shape_Classification",
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
    class_weight: str | None,
    fewshot_shots: list[int],
    seeds: list[int],
    experiment_configs,
    training_config,
    output_path: Path | None = None,
    target_sensors: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    subset = metadata[metadata["sensor"].isin(sensors)].reset_index(drop=True)
    num_classes = int(subset["label"].nunique())
    eval_targets = target_sensors if target_sensors else sensors

    if any(config.init_mode == "mae" for config in experiment_configs):
        ensure_mae_checkpoint(training_config)

    for seed in seeds:
        seeded_training_config = with_seed(training_config, seed)
        for config in experiment_configs:
            print(f"Running shape {variant} for {config.model_name} / {config.init_mode} / seed {seed}")

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
                        model_name=config.model_name,
                        init_mode_requested=config.init_mode,
                        class_weight=class_weight,
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
                            model_name=config.model_name,
                            init_mode_requested=config.init_mode,
                            class_weight=class_weight,
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
                        model_name=config.model_name,
                        init_mode_requested=config.init_mode,
                        class_weight=class_weight,
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
                            label_col="label",
                            num_units_per_label=shots,
                            unit_col="image_path",
                            sort_cols=["label", "sample_id"],
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
                            model_name=config.model_name,
                            init_mode_requested=config.init_mode,
                            class_weight=class_weight,
                            fewshot_value=shots,
                            fewshot_unit="shots_per_class",
                            support_units=int(len(support_frame)),
                            support_samples=int(len(support_frame)),
                            num_classes=num_classes,
                        )

            if output_path is not None and rows:
                pd.DataFrame(rows).to_csv(output_path, index=False)

    return pd.DataFrame(rows)


def print_metadata_summary(metadata: pd.DataFrame, balanced_sensors: list[str]) -> None:
    counts = metadata.groupby(["sensor", "label"]).size().unstack(fill_value=0).sort_index()
    print(counts)
    print(f"Balanced sensors: {', '.join(balanced_sensors)}")


def main() -> None:
    args = parse_args()
    metadata = build_shape_metadata()

    if args.sensors:
        sensor_subset = [sensor.strip() for sensor in args.sensors.split(",") if sensor.strip()]
        metadata = metadata[metadata["sensor"].isin(sensor_subset)].reset_index(drop=True)

    if metadata.empty:
        raise RuntimeError("No shape classification samples were found.")

    balanced_sensors = infer_balanced_sensors(metadata)
    print_metadata_summary(metadata, balanced_sensors)
    if args.metadata_only:
        return

    protocols = {protocol.strip() for protocol in args.protocols.split(",") if protocol.strip()}
    variants = {variant.strip() for variant in args.variants.split(",") if variant.strip()}
    fewshot_shots = parse_int_list(args.fewshot_shots)
    seeds = parse_seed_list(args.seeds, args.seed)
    experiment_configs = build_experiment_configs(args.models, args.vit_init_modes)
    training_config = build_training_config(args)
    target_sensors = [s.strip() for s in args.target_sensors.split(",") if s.strip()] or None

    all_rows: list[pd.DataFrame] = []
    if "balanced" in variants and balanced_sensors:
        all_rows.append(
            run_variant(
                metadata=metadata,
                sensors=balanced_sensors,
                protocols=protocols,
                variant="balanced",
                class_weight=None,
                fewshot_shots=fewshot_shots,
                seeds=seeds,
                experiment_configs=experiment_configs,
                training_config=training_config,
                output_path=args.output,
                target_sensors=target_sensors,
            )
        )
    if "longtail" in variants:
        all_rows.append(
            run_variant(
                metadata=metadata,
                sensors=sorted(metadata["sensor"].unique()),
                protocols=protocols,
                variant="longtail",
                class_weight="balanced",
                fewshot_shots=fewshot_shots,
                seeds=seeds,
                experiment_configs=experiment_configs,
                training_config=training_config,
                output_path=args.output,
                target_sensors=target_sensors,
            )
        )

    if not all_rows:
        raise RuntimeError("No shape experiments were produced. Check sensor and variant filters.")

    results = pd.concat(all_rows, ignore_index=True)
    results.insert(0, "image_size", args.image_size)
    ensure_results_dir()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Saved {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
