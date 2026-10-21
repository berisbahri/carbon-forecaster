import numpy as np
import pandas as pd

from src.features import CALENDAR_COLS, LAG_COLS, build_features, calendar_features

SLOTS = 48


def make_intensity(days: int = 14) -> pd.DataFrame:
    idx = pd.date_range("2025-03-01", periods=days * SLOTS, freq="30min", tz="UTC")
    actual = np.arange(len(idx), dtype=float)
    return pd.DataFrame({"actual": actual, "forecast": actual}, index=idx)


def test_no_leakage_from_the_last_24_hours():
    """Changing anything within 24h of the target must not change its features."""
    df = make_intensity()
    target = df.index[-1]
    before = build_features(df).loc[target, LAG_COLS]

    tampered = df.copy()
    recent = tampered.index > target - pd.Timedelta(hours=24)
    tampered.loc[recent, "actual"] = -999.0
    after = build_features(tampered).loc[target, LAG_COLS]

    pd.testing.assert_series_equal(before, after)


def test_lag_24h_is_value_one_day_earlier():
    df = make_intensity()
    feats = build_features(df)
    t = df.index[-1]
    assert feats.loc[t, "lag_24h"] == df.loc[t - pd.Timedelta(hours=24), "actual"]


def test_calendar_uses_uk_local_time():
    # 12:00 UTC in July is 13:00 BST -> slot 26
    idx = pd.DatetimeIndex([pd.Timestamp("2025-07-01 12:00", tz="UTC")])
    assert calendar_features(idx).loc[idx[0], "slot"] == 26


def test_build_features_has_all_columns():
    feats = build_features(make_intensity())
    for col in ["actual", *LAG_COLS, *CALENDAR_COLS]:
        assert col in feats.columns
