import numpy as np
import pandas as pd
import pytest

from src.evaluate import SLOTS_PER_DAY, add_baselines, evaluate, metrics, time_split


def make_df(days: int = 20) -> pd.DataFrame:
    """Synthetic half-hourly data: actual = 0, 1, 2, ... so shifts are easy to check."""
    idx = pd.date_range("2025-01-01", periods=days * SLOTS_PER_DAY, freq="30min", tz="UTC")
    actual = np.arange(len(idx), dtype=float)
    return pd.DataFrame({"actual": actual, "forecast": actual + 1}, index=idx)


def test_time_split_has_no_overlap():
    df = make_df(30)
    train, test, cutoff = time_split(df, test_days=7)
    assert train.index.max() <= cutoff < test.index.min()
    assert len(train) + len(test) == len(df)


def test_yesterday_baseline_is_exactly_24h_earlier():
    df = add_baselines(make_df())
    t = df.index[100]
    assert df.loc[t, "pred_yesterday"] == df.loc[t - pd.Timedelta(hours=24), "actual"]


def test_last_week_baseline_is_exactly_7_days_earlier():
    df = add_baselines(make_df())
    t = df.index[-1]
    assert df.loc[t, "pred_last_week"] == df.loc[t - pd.Timedelta(days=7), "actual"]


def test_metrics_known_values():
    m = metrics(np.array([10, 20, 30]), np.array([12, 18, 30]))
    assert m["mae"] == pytest.approx(4 / 3)
    assert m["bias"] == pytest.approx(0)
    assert m["n"] == 3


def test_evaluate_scores_all_methods_on_the_same_rows():
    df = add_baselines(make_df())
    df.iloc[-5:, df.columns.get_loc("pred_national_grid")] = np.nan
    table = evaluate(df, ["pred_yesterday", "pred_last_week", "pred_national_grid"])
    assert table["n"].nunique() == 1