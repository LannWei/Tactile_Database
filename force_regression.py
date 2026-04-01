from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    PROJECT_ROOT,
    assign_ordered_split,
    ensure_results_dir,
    extract_numeric_suffix,
    format_sensor_list,
    regression_metrics,
    select_first_n_units,
)
from torch_common import (
    add_torch_args,
    build_experiment_configs,
    build_training_config,
    ensure_mae_checkpoint,
    experiment_label,
    parse_seed_list,
    train_regression_experiment,
    with_seed,
)


FORCE_DIR = PROJECT_ROOT / "Force_Regression"
AXES = ("fx", "fy", "fz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run force regression benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "force_regression_results.csv",
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
        "--fewshot-ratios",
        type=str,
        default="0.005,0.01,0.05,0.1",
        help="Comma-separated target-train ratios used for few-shot adaptation.",
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


def build_force_metadata() -> pd.DataFrame:
    rows: list[dict] = []

    for sensor_dir in sorted(FORCE_DIR.iterdir()):
        if not sensor_dir.is_dir():
            continue

        csv_path = sensor_dir / f"{sensor_dir.name}.csv"
        if not csv_path.exists():
            continue

        csv_data = pd.read_csv(csv_path)
        csv_data.columns = [column.strip() for column in csv_data.columns]

        for record in csv_data.itertuples(index=False):
            image_name = str(record.image_name).strip()
            image_path = sensor_dir / image_name
            if not image_path.exists():
                continue

            sample_id = extract_numeric_suffix(Path(image_name).stem)
            rows.append(
                {
                    "sensor": sensor_dir.name,
                    "image_path": str(image_path),
                    "image_name": image_name,
                    "sample_id": sample_id,
                    "fx": float(record.fx),
                    "fy": float(record.fy),
                    "fz": float(record.fz),
                }
            )

    metadata = pd.DataFrame(rows).sort_values(["sensor", "sample_id"]).reset_index(drop=True)
    return assign_ordered_split(metadata, group_cols=["sensor"], order_col="sample_id")


def parse_float_list(raw_value: str) -> list[float]:
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def fit_and_score(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    model_name: str,
    init_mode: str,
    training_config,
) -> dict[str, float]:
    outputs = train_regression_experiment(
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        target_cols=AXES,
        model_name=model_name,
        init_mode=init_mode,
        config=training_config,
    )
    metrics = regression_metrics(outputs["y_true"], outputs["y_pred"], AXES)
    return {
        **metrics,
        "init_mode": outputs["resolved_init_mode"],
        "device": outputs["device"],
        "best_epoch": outputs["best_epoch"],
    }


def append_row(
    rows: list[dict],
    training_config,
    protocol: str,
    source_sensors: str,
    target_sensor: str,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    model_name: str,
    init_mode_requested: str,
    fewshot_value,
    fewshot_unit: str,
    support_units: int,
    support_samples: int,
) -> None:
    metrics = fit_and_score(
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        model_name=model_name,
        init_mode=init_mode_requested,
        training_config=training_config,
    )
    rows.append(
        {
            "task": "Force_Regression",
            "seed": training_config.seed,
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
            "train_samples": int(len(train_frame)),
            "val_samples": int(len(val_frame)),
            "test_samples": int(len(test_frame)),
            **metrics,
        }
    )
    row = rows[-1]
    print(
        f"  [{len(rows):>4d}] {protocol:<22s} {source_sensors:<30s} → {target_sensor:<20s} "
        f"RMSE={row.get('rmse', 0):.4f}  MAE={row.get('mae', 0):.4f}  "
        f"epoch={row['best_epoch']}"
    )


def run_experiments(
    metadata: pd.DataFrame,
    sensors: list[str],
    protocols: set[str],
    fewshot_ratios: list[float],
    seeds: list[int],
    experiment_configs,
    training_config,
    output_path: Path | None = None,
    target_sensors: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    eval_targets = target_sensors if target_sensors else sensors

    if any(config.init_mode == "mae" for config in experiment_configs):
        ensure_mae_checkpoint(training_config)

    for seed in seeds:
        seeded_training_config = with_seed(training_config, seed)
        for config in experiment_configs:
            print(f"Running force experiments for {config.model_name} / {config.init_mode} / seed {seed}")

            if "within" in protocols:
                for sensor in eval_targets:
                    train_frame = metadata[
                        metadata["sensor"].eq(sensor) & metadata["split"].eq("train")
                    ].reset_index(drop=True)
                    val_frame = metadata[
                        metadata["sensor"].eq(sensor) & metadata["split"].eq("val")
                    ].reset_index(drop=True)
                    test_frame = metadata[
                        metadata["sensor"].eq(sensor) & metadata["split"].eq("test")
                    ].reset_index(drop=True)
                    if train_frame.empty or test_frame.empty:
                        continue
                    append_row(
                        rows=rows,
                        training_config=seeded_training_config,
                        protocol="within_sensor",
                        source_sensors=sensor,
                        target_sensor=sensor,
                        train_frame=train_frame,
                        val_frame=val_frame,
                        test_frame=test_frame,
                        model_name=config.model_name,
                        init_mode_requested=config.init_mode,
                        fewshot_value=np.nan,
                        fewshot_unit="",
                        support_units=0,
                        support_samples=0,
                    )

            if "cross" in protocols:
                for source_sensor in sensors:
                    for target_sensor in eval_targets:
                        if source_sensor == target_sensor:
                            continue
                        train_frame = metadata[
                            metadata["sensor"].eq(source_sensor) & metadata["split"].eq("train")
                        ].reset_index(drop=True)
                        val_frame = metadata[
                            metadata["sensor"].eq(source_sensor) & metadata["split"].eq("val")
                        ].reset_index(drop=True)
                        test_frame = metadata[
                            metadata["sensor"].eq(target_sensor) & metadata["split"].eq("test")
                        ].reset_index(drop=True)
                        if train_frame.empty or test_frame.empty:
                            continue
                        append_row(
                            rows=rows,
                            training_config=seeded_training_config,
                            protocol="cross_sensor",
                            source_sensors=source_sensor,
                            target_sensor=target_sensor,
                            train_frame=train_frame,
                            val_frame=val_frame,
                            test_frame=test_frame,
                            model_name=config.model_name,
                            init_mode_requested=config.init_mode,
                            fewshot_value=np.nan,
                            fewshot_unit="",
                            support_units=0,
                            support_samples=0,
                        )

            if "loso" in protocols:
                for target_sensor in eval_targets:
                    source_sensors = [sensor for sensor in sensors if sensor != target_sensor]
                    train_frame = metadata[
                        metadata["sensor"].isin(source_sensors) & metadata["split"].eq("train")
                    ].reset_index(drop=True)
                    val_frame = metadata[
                        metadata["sensor"].isin(source_sensors) & metadata["split"].eq("val")
                    ].reset_index(drop=True)
                    test_frame = metadata[
                        metadata["sensor"].eq(target_sensor) & metadata["split"].eq("test")
                    ].reset_index(drop=True)
                    if train_frame.empty or test_frame.empty:
                        continue
                    append_row(
                        rows=rows,
                        training_config=seeded_training_config,
                        protocol="leave_one_sensor_out",
                        source_sensors=format_sensor_list(source_sensors),
                        target_sensor=target_sensor,
                        train_frame=train_frame,
                        val_frame=val_frame,
                        test_frame=test_frame,
                        model_name=config.model_name,
                        init_mode_requested=config.init_mode,
                        fewshot_value=np.nan,
                        fewshot_unit="",
                        support_units=0,
                        support_samples=0,
                    )

            if "fewshot" in protocols:
                for target_sensor in eval_targets:
                    source_sensors = [sensor for sensor in sensors if sensor != target_sensor]
                    source_train = metadata[
                        metadata["sensor"].isin(source_sensors) & metadata["split"].eq("train")
                    ]
                    source_val = metadata[
                        metadata["sensor"].isin(source_sensors) & metadata["split"].eq("val")
                    ]
                    target_support_pool = metadata[
                        metadata["sensor"].eq(target_sensor) & metadata["split"].eq("train")
                    ]
                    test_frame = metadata[
                        metadata["sensor"].eq(target_sensor) & metadata["split"].eq("test")
                    ].reset_index(drop=True)
                    support_pool_size = int(len(target_support_pool))

                    if source_train.empty or test_frame.empty or support_pool_size == 0:
                        continue

                    for ratio in fewshot_ratios:
                        support_units = max(1, int(round(support_pool_size * ratio)))
                        eligible_mask = pd.Series(False, index=metadata.index)
                        eligible_mask.loc[target_support_pool.index] = True
                        support_mask = select_first_n_units(
                            metadata=metadata,
                            eligible_mask=eligible_mask,
                            num_units=support_units,
                            unit_col="image_name",
                            sort_cols=["sample_id"],
                            seed=seed,
                        )
                        support_frame = metadata[support_mask].reset_index(drop=True)
                        train_frame = pd.concat([source_train, support_frame], ignore_index=True)
                        append_row(
                            rows=rows,
                            training_config=seeded_training_config,
                            protocol="fewshot_adaptation",
                            source_sensors=format_sensor_list(source_sensors),
                            target_sensor=target_sensor,
                            train_frame=train_frame,
                            val_frame=source_val.reset_index(drop=True),
                            test_frame=test_frame,
                            model_name=config.model_name,
                            init_mode_requested=config.init_mode,
                            fewshot_value=ratio,
                            fewshot_unit="target_train_ratio",
                            support_units=int(len(support_frame)),
                            support_samples=int(len(support_frame)),
                        )

            if output_path is not None and rows:
                pd.DataFrame(rows).to_csv(output_path, index=False)

    return pd.DataFrame(rows)


def print_metadata_summary(metadata: pd.DataFrame) -> None:
    summary = metadata.groupby(["sensor", "split"]).size().unstack(fill_value=0).sort_index()
    print(summary)


def main() -> None:
    args = parse_args()
    metadata = build_force_metadata()

    if args.sensors:
        sensor_subset = [sensor.strip() for sensor in args.sensors.split(",") if sensor.strip()]
        metadata = metadata[metadata["sensor"].isin(sensor_subset)].reset_index(drop=True)

    if metadata.empty:
        raise RuntimeError("No force regression samples were found.")

    print_metadata_summary(metadata)
    if args.metadata_only:
        return

    sensors = sorted(metadata["sensor"].unique())
    protocols = {protocol.strip() for protocol in args.protocols.split(",") if protocol.strip()}
    fewshot_ratios = parse_float_list(args.fewshot_ratios)
    seeds = parse_seed_list(args.seeds, args.seed)
    experiment_configs = build_experiment_configs(args.models, args.vit_init_modes)
    training_config = build_training_config(args)
    target_sensors = [s.strip() for s in args.target_sensors.split(",") if s.strip()] or None

    results = run_experiments(
        metadata=metadata,
        sensors=sensors,
        protocols=protocols,
        fewshot_ratios=fewshot_ratios,
        seeds=seeds,
        experiment_configs=experiment_configs,
        training_config=training_config,
        output_path=args.output,
        target_sensors=target_sensors,
    )
    results.insert(0, "image_size", args.image_size)

    ensure_results_dir()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Saved {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
