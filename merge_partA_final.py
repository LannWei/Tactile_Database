"""Final merge: combine existing Part A data with newly-run GPU results.

Run this AFTER resume_partA.sh completes on GPU.

Merges:
  Force:   partA_force.csv (8 rows from merge_partA.py) + partA_force_fewshot.csv (4 new)
  Shape:   partA_shape.csv (10 rows from merge_partA.py) + partA_shape_fewshot.csv (2 new)
  Grating: partA_grating.csv (1 within from Part B) + partA_grating_remaining.csv (7 new)
"""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def safe_load(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        print(f"  Loaded {len(df)} rows from {path.name}")
        return df
    print(f"  WARNING: {path.name} not found")
    return pd.DataFrame()


def merge_and_save(name: str, *csv_names: str) -> None:
    print(f"\n── {name} ──")
    parts = []
    for csv_name in csv_names:
        df = safe_load(RESULTS_DIR / csv_name)
        if not df.empty:
            parts.append(df)

    if not parts:
        print(f"  No data to merge for {name}")
        return

    merged = pd.concat(parts, ignore_index=True)
    out_path = RESULTS_DIR / f"partA_{name}.csv"
    merged.to_csv(out_path, index=False)
    print(f"  → Saved {len(merged)} rows to {out_path.name}")


def main():
    print("=" * 60)
    print("Part A — Final merge")
    print("=" * 60)

    merge_and_save("force", "partA_force.csv", "partA_force_fewshot.csv")
    merge_and_save("shape", "partA_shape.csv", "partA_shape_fewshot.csv")
    merge_and_save("grating", "partA_grating.csv", "partA_grating_remaining.csv")

    # Summary
    print("\n" + "=" * 60)
    expected = {"force": 12, "shape": 12, "grating": 8}
    for task, n in expected.items():
        path = RESULTS_DIR / f"partA_{task}.csv"
        if path.exists():
            actual = len(pd.read_csv(path))
            status = "✓" if actual == n else f"⚠ ({actual}/{n})"
            print(f"  {status} partA_{task}.csv: {actual} rows (expected {n})")
        else:
            print(f"  ✗ partA_{task}.csv: MISSING")


if __name__ == "__main__":
    main()
