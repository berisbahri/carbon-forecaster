"""
Phase 2: evaluation harness and baselines.

  - Loads data/intensity.csv onto a regular half-hourly grid
  - Splits by time: everything before the cutoff is train, the last
    TEST_DAYS days are test (never a random split)
  - Scores the baselines and National Grid's forecast on the test set,
    all on exactly the same rows so the comparison is fair
  - Saves the results table to data/metrics.json

Usage (from the project root, with the venv active):
    python -m src.evaluate

Phase 3 reuses load_intensity, time_split and evaluate: add a "pred_model"
column to the test set and pass it in alongside the baselines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SLOTS_PER_DAY = 48  # half-hourly data
TEST_DAYS = 90


def load_intensity(path: Path = DATA_DIR / "intensity.csv") -> pd.DataFrame:
    """Load intensity data indexed by time on a complete 30-minute grid.

    Missing half-hours become NaN rows, so that shifting by 48 rows always
    means "exactly 24 hours earlier", even where the API had gaps.
    """
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="30min")
    df = df.reindex(full_index)
    df.index.name = "time"

    # trailing periods without an actual haven't been measured yet
    last_actual = df["actual"].last_valid_index()
    return df.loc[:last_actual]


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Add naive forecasts. Each uses only data available 24h+ beforehand."""
    out = df.copy()
    out["pred_yesterday"] = out["actual"].shift(SLOTS_PER_DAY)
    out["pred_last_week"] = out["actual"].shift(7 * SLOTS_PER_DAY)
    out["pred_national_grid"] = out["forecast"]
    return out


def time_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    """Train = everything up to the cutoff, test = the last `test_days` days."""
    cutoff = df.index.max() - pd.Timedelta(days=test_days)
    train = df.loc[df.index <= cutoff]
    test = df.loc[df.index > cutoff]
    return train, test, cutoff


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),  # positive = over-predicting
        "n": int(len(err)),
    }


def evaluate(test: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    """Score each prediction column against `actual` on the SAME rows.

    Rows where the actual or any prediction is missing are dropped for
    every method, otherwise one method could be scored on easier rows.
    """
    common = test.dropna(subset=["actual", *pred_cols])
    if common.empty:
        raise ValueError("No rows where every prediction and the actual exist.")

    results = {
        col.removeprefix("pred_"): metrics(common["actual"], common[col])
        for col in pred_cols
    }
    table = pd.DataFrame(results).T
    table["n"] = table["n"].astype(int)

    if "yesterday" in table.index:
        ref = table.loc["yesterday", "mae"]
        table["skill_vs_yesterday_%"] = 100 * (1 - table["mae"] / ref)

    return table.sort_values("mae")


def data_quality_report(df: pd.DataFrame) -> None:
    n = len(df)
    missing_actual = df["actual"].isna().sum()
    missing_forecast = df["forecast"].isna().sum()
    print(f"Data: {n:,} half-hours, {df.index.min()} -> {df.index.max()}")
    print(f"  missing actual:   {missing_actual:,} ({100 * missing_actual / n:.2f}%)")
    print(f"  missing forecast: {missing_forecast:,} ({100 * missing_forecast / n:.2f}%)")
    print(f"  actual range: {df['actual'].min():.0f} to {df['actual'].max():.0f} gCO2/kWh, "
          f"mean {df['actual'].mean():.0f}")


def main() -> None:
    df = load_intensity()
    data_quality_report(df)

    df = add_baselines(df)
    train, test, cutoff = time_split(df)
    print(f"\nTrain: {train.index.min()} -> {cutoff}  ({len(train):,} rows)")
    print(f"Test:  {test.index.min()} -> {test.index.max()}  ({len(test):,} rows)")

    pred_cols = ["pred_yesterday", "pred_last_week", "pred_national_grid"]
    table = evaluate(test, pred_cols)

    print("\nTest-set results (gCO2/kWh):")
    print(table.round(2).to_string())
    print("\nNote: National Grid's stored forecast is made close to each period,"
          "\nnot 24h ahead, so it is a tougher benchmark than our 24h-ahead model faces.")

    DATA_DIR.mkdir(exist_ok=True)
    out = {
        "cutoff": cutoff.isoformat(),
        "test_start": test.index.min().isoformat(),
        "test_end": test.index.max().isoformat(),
        "results": table.round(3).to_dict(orient="index"),
    }
    (DATA_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {DATA_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()