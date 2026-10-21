"""
Phase 5: forecast service behind the API (no web framework in here).

Keeps the latest forecast in memory and refreshes it at most every
MAX_AGE, so API calls are instant and the free upstream APIs aren't hit
on every request. If a refresh fails it serves the last good forecast
(marked stale) rather than erroring, falling back to
data/latest_forecast.csv after a restart.
"""
from __future__ import annotations

import logging
import math
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from src.evaluate import DATA_DIR

log = logging.getLogger(__name__)

MAX_AGE = timedelta(minutes=30)
UNIT = "gCO2/kWh"


class ForecastUnavailable(Exception):
    pass


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class ForecastService:
    def __init__(self,
                 forecaster: Callable[[pd.Timestamp], pd.DataFrame] | None = None,
                 on_refresh: Callable[[pd.DataFrame, pd.Timestamp], None] | None = None,
                 max_age: timedelta = MAX_AGE,
                 clock: Callable[[], pd.Timestamp] = _now,
                 fallback_path: Path = DATA_DIR / "latest_forecast.csv"):
        if forecaster is None or on_refresh is None:
            from src.predict import make_forecast, save  # real network calls
            forecaster = forecaster or make_forecast
            on_refresh = on_refresh or save
        self._forecaster = forecaster
        self._on_refresh = on_refresh
        self._max_age = max_age
        self._clock = clock
        self._fallback_path = fallback_path
        self._lock = threading.Lock()
        self._forecast: pd.DataFrame | None = None
        self._issued_at: pd.Timestamp | None = None

    def get(self) -> tuple[pd.DataFrame, pd.Timestamp, bool]:
        """Return (forecast, issued_at, stale), refreshing if too old."""
        with self._lock:
            now = self._clock()
            fresh = (self._forecast is not None
                     and now - self._issued_at < self._max_age)
            stale = False
            if not fresh:
                try:
                    forecast = self._forecaster(now)
                    self._forecast, self._issued_at = forecast, now
                    try:
                        self._on_refresh(forecast, now)
                    except Exception:  # saving is nice-to-have, never fatal
                        log.exception("could not save forecast")
                except Exception as e:
                    log.warning("forecast refresh failed: %s", e)
                    stale = True
                    if self._forecast is None:
                        self._load_fallback()
                    if self._forecast is None:
                        raise ForecastUnavailable(str(e)) from e

            upcoming = self._forecast[self._forecast.index >= now.floor("30min")]
            if upcoming.empty:
                raise ForecastUnavailable("cached forecast has expired")
            return upcoming, self._issued_at, stale

    def _load_fallback(self) -> None:
        if not self._fallback_path.exists():
            return
        df = pd.read_csv(self._fallback_path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        self._forecast = df.set_index("time")
        mtime = self._fallback_path.stat().st_mtime
        self._issued_at = pd.Timestamp(mtime, unit="s", tz="UTC")


def parse_deadline(value: datetime | str | None) -> pd.Timestamp | None:
    """Times without a timezone are read as UK local time."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Europe/London")
    return ts.tz_convert("UTC")


def _num(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, 1)


def forecast_to_json(forecast: pd.DataFrame, issued_at: pd.Timestamp, stale: bool) -> dict:
    ng = forecast["national_grid"] if "national_grid" in forecast else None
    return {
        "region": "national",
        "unit": UNIT,
        "issued_at": issued_at.isoformat(),
        "stale": stale,
        "points": [
            {"time": t.isoformat(),
             "pred": _num(forecast.at[t, "pred"]),
             "national_grid": _num(ng[t]) if ng is not None else None}
            for t in forecast.index
        ],
    }


def best_to_json(best: dict, issued_at: pd.Timestamp, stale: bool) -> dict:
    return {
        "start": best["start"].isoformat(),
        "end": best["end"].isoformat(),
        "hours": best["hours"],
        "avg_intensity": _num(best["avg_intensity"]),
        "start_now_avg": _num(best["start_now_avg"]),
        "saving_pct": _num(best["saving_pct"]),
        "unit": UNIT,
        "issued_at": issued_at.isoformat(),
        "stale": stale,
    }
