from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def extract_numeric_suffix(name: str) -> int:
    digits = []
    for char in reversed(name):
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    if not digits:
        raise ValueError(f"Could not parse numeric suffix from: {name}")
    return int("".join(reversed(digits)))


def assign_ordered_split(
    metadata: pd.DataFrame,
    group_cols: Sequence[str],
    order_col: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> pd.DataFrame:
    annotated = metadata.copy()
    split_values = pd.Series(index=annotated.index, dtype="object")

    for _, group in annotated.groupby(list(group_cols), sort=False):
        ordered_ids = np.array(sorted(group[order_col].unique()))
        num_unique = len(ordered_ids)
        train_end = max(1, int(round(num_unique * train_ratio)))
        val_end = max(train_end + 1, int(round(num_unique * (train_ratio + val_ratio))))
        val_end = min(val_end, num_unique)

        train_ids = set(ordered_ids[:train_end])
        val_ids = set(ordered_ids[train_end:val_end])

        split_values.loc[group.index] = [
            "train"
            if value in train_ids
            else "val"
            if value in val_ids
            else "test"
            for value in group[order_col]
        ]

    annotated["split"] = split_values
    return annotated


def classification_metrics(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    true_array = np.asarray(list(y_true))
    pred_array = np.asarray(list(y_pred))
    labels = np.unique(np.concatenate([true_array, pred_array]))

    accuracy = float(np.mean(true_array == pred_array))

    recalls: list[float] = []
    f1_scores: list[float] = []
    for label in labels:
        true_positive = np.sum((true_array == label) & (pred_array == label))
        false_positive = np.sum((true_array != label) & (pred_array == label))
        false_negative = np.sum((true_array == label) & (pred_array != label))
        actual_positive = np.sum(true_array == label)

        recall = 0.0 if actual_positive == 0 else float(true_positive / actual_positive)
        precision_denom = true_positive + false_positive
        precision = 0.0 if precision_denom == 0 else float(true_positive / precision_denom)
        f1_denom = precision + recall
        f1_value = 0.0 if f1_denom == 0 else float(2.0 * precision * recall / f1_denom)

        recalls.append(recall)
        f1_scores.append(f1_value)

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis_names: Sequence[str],
) -> dict[str, float]:
    errors = y_pred - y_true
    absolute = np.abs(errors)
    squared = errors**2

    results: dict[str, float] = {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(squared.mean())),
    }

    pearsons: list[float] = []
    for axis_index, axis_name in enumerate(axis_names):
        true_axis = y_true[:, axis_index]
        pred_axis = y_pred[:, axis_index]
        results[f"mae_{axis_name}"] = float(np.mean(np.abs(pred_axis - true_axis)))
        results[f"rmse_{axis_name}"] = float(
            np.sqrt(np.mean((pred_axis - true_axis) ** 2))
        )

        true_std = float(np.std(true_axis))
        pred_std = float(np.std(pred_axis))
        if true_std == 0.0 or pred_std == 0.0:
            pearson = np.nan
        else:
            pearson = float(np.corrcoef(true_axis, pred_axis)[0, 1])
        results[f"pearson_{axis_name}"] = pearson
        if not np.isnan(pearson):
            pearsons.append(pearson)

    results["pearson_mean"] = float(np.mean(pearsons)) if pearsons else np.nan
    return results


def format_sensor_list(sensors: Iterable[str]) -> str:
    return ";".join(sorted(set(sensors)))


def size_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true_array = np.asarray(y_true, dtype=np.float32)
    pred_array = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(pred_array - true_array)))


def select_first_n_units(
    metadata: pd.DataFrame,
    eligible_mask: pd.Series,
    num_units: int,
    unit_col: str,
    sort_cols: Sequence[str],
    seed: int | None = None,
) -> pd.Series:
    if num_units <= 0:
        return pd.Series(False, index=metadata.index)

    selected_cols: list[str] = []
    for column in [unit_col, *sort_cols]:
        if column not in selected_cols:
            selected_cols.append(column)
    eligible = metadata.loc[eligible_mask, selected_cols].copy()

    sort_order: list[str] = []
    for column in [*sort_cols, unit_col]:
        if column not in sort_order:
            sort_order.append(column)
    ordered = eligible.sort_values(sort_order).drop_duplicates(unit_col)
    unit_values = ordered[unit_col].to_numpy()
    if len(unit_values) <= num_units or seed is None:
        selected_units = unit_values[:num_units]
    else:
        rng = np.random.default_rng(seed)
        selected_units = rng.permutation(unit_values)[:num_units]
    return eligible_mask & metadata[unit_col].isin(selected_units)


def select_fewshot_by_label(
    metadata: pd.DataFrame,
    eligible_mask: pd.Series,
    label_col: str,
    num_units_per_label: int,
    unit_col: str,
    sort_cols: Sequence[str],
    seed: int | None = None,
) -> pd.Series:
    if num_units_per_label <= 0:
        return pd.Series(False, index=metadata.index)

    selected_cols: list[str] = []
    for column in [label_col, unit_col, *sort_cols]:
        if column not in selected_cols:
            selected_cols.append(column)
    eligible = metadata.loc[eligible_mask, selected_cols].copy()

    sort_order: list[str] = []
    for column in [label_col, *sort_cols, unit_col]:
        if column not in sort_order:
            sort_order.append(column)
    ordered = eligible.sort_values(sort_order).drop_duplicates(unit_col)
    rng = np.random.default_rng(seed) if seed is not None else None
    selected_units: list = []
    for _, group in ordered.groupby(label_col, sort=False):
        unit_values = group[unit_col].to_numpy()
        sample_size = min(num_units_per_label, len(unit_values))
        if rng is None or len(unit_values) <= sample_size:
            chosen = unit_values[:sample_size]
        else:
            chosen = rng.permutation(unit_values)[:sample_size]
        selected_units.extend(chosen.tolist())
    return eligible_mask & metadata[unit_col].isin(selected_units)
