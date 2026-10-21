"""
Phase 3: feature engineering for a 24-hour-ahead forecast.

The forecast is made at the "origin" time, 24 hours before the target
half-hour. Every lag and rolling feature below is therefore built only from
data at least 24 hours older than the target, so nothing from the future
can leak in. tests/test_features.py checks this directly.

Feature groups:
  - lags:     intensity 24h, 48h and 7 days before the target, plus rolling
              means over the day and week leading up to the origin
  - calendar: half-hour slot, day of week, month, weekend flag and a smooth
              time-of-year signal, all in UK local time (demand follows
              clocks, including BST)
  - weather:  wind at 100 m, solar radiation, temperature and cloud cover
              at the target time (see the "oracle weather" note in train.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SLOTS_PER_DAY = 48
HORIZON_SLOTS = SLOTS_PER_DAY  # 24 hours ahead

WEATHER_COLS = ["wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover"]
LAG_COLS = ["lag_24h", "lag_48h", "lag_168h", "mean_24h_at_origin", "mean_7d_at_origin"]
CALENDAR_COLS = ["slot", "dayofweek", "month", "is_weekend", "doy_sin", "doy_cos"]

FEATURES = LAG_COLS + CALENDAR_COLS + WEATHER_COLS
FEATURES_NO_WEATHER = LAG_COLS + CALENDAR_COLS


def load_weather(path: Path = DATA_DIR / "weather.csv") -> pd.DataFrame:
    """Hourly national weather, upsampled to the half-hourly grid."""
    w = pd.read_csv(path)
    w["time"] = pd.to_datetime(w["time"], utc=True)
    w = w.drop_duplicates("time").set_index("time").sort_index()[WEATHER_COLS]
    # fill the :30 slots between hourly readings by linear interpolation
    return w.resample("30min").interpolate(limit=1)


def lag_features(actual: pd.Series) -> pd.DataFrame:
    """Lag features. `actual` must be on a complete 30-minute grid."""
    at_origin = actual.shift(HORIZON_SLOTS)  # the latest value known when forecasting
    return pd.DataFrame({
        "lag_24h": at_origin,
        "lag_48h": actual.shift(2 * SLOTS_PER_DAY),
        "lag_168h": actual.shift(7 * SLOTS_PER_DAY),
        "mean_24h_at_origin": at_origin.rolling(SLOTS_PER_DAY, min_periods=36).mean(),
        "mean_7d_at_origin": at_origin.rolling(7 * SLOTS_PER_DAY, min_periods=250).mean(),
    }, index=actual.index)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert("Europe/London")
    doy = local.dayofyear.to_numpy()
    return pd.DataFrame({
        "slot": local.hour * 2 + local.minute // 30,
        "dayofweek": local.dayofweek,
        "month": local.month,
        "is_weekend": (local.dayofweek >= 5).astype(int),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=index)


def build_features(intensity: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return `actual` (the target) plus every feature, indexed by target time.

    `intensity` should come from evaluate.load_intensity(), which guarantees
    the complete half-hourly grid that the lag features rely on.
    """
    parts = [intensity[["actual"]], lag_features(intensity["actual"]),
             calendar_features(intensity.index)]
    if weather is not None:
        parts.append(weather.reindex(intensity.index))
    return pd.concat(parts, axis=1)