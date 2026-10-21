from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.service import (ForecastService, ForecastUnavailable, best_to_json,
                         forecast_to_json, parse_deadline)
from src.best_time import best_window

T0 = pd.Timestamp("2026-10-21 14:10", tz="UTC")


def fake_forecast(now: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(now.floor("30min") + pd.Timedelta(minutes=30),
                        periods=48, freq="30min")
    return pd.DataFrame({"pred": np.linspace(200, 100, 48),
                         "national_grid": np.linspace(190, 110, 48)},
                        index=idx)


class Clock:
    def __init__(self, t): self.t = t
    def __call__(self): return self.t


def make_service(clock, forecaster=fake_forecast):
    return ForecastService(forecaster=forecaster, on_refresh=lambda *a: None,
                           max_age=timedelta(minutes=30), clock=clock,
                           fallback_path=Path("does-not-exist/latest_forecast.csv"))


def test_caches_within_max_age():
    calls = []
    def counting(now):
        calls.append(now)
        return fake_forecast(now)
    clock = Clock(T0)
    svc = make_service(clock, counting)
    svc.get()
    clock.t = T0 + pd.Timedelta(minutes=10)
    svc.get()
    assert len(calls) == 1
    clock.t = T0 + pd.Timedelta(minutes=31)
    svc.get()
    assert len(calls) == 2


def test_serves_stale_forecast_when_refresh_fails():
    clock = Clock(T0)
    state = {"fail": False}
    def flaky(now):
        if state["fail"]:
            raise RuntimeError("upstream down")
        return fake_forecast(now)
    svc = make_service(clock, flaky)
    svc.get()
    state["fail"] = True
    clock.t = T0 + pd.Timedelta(hours=2)
    fc, _, stale = svc.get()
    assert stale
    assert fc.index.min() >= clock.t.floor("30min")  # past slots dropped


def test_raises_when_nothing_available():
    def broken(now):
        raise RuntimeError("no network")
    with pytest.raises(ForecastUnavailable):
        make_service(Clock(T0), broken).get()


def test_parse_deadline_reads_naive_times_as_uk_time():
    # 07:00 BST in October = 06:00 UTC
    assert parse_deadline("2026-10-22T07:00") == pd.Timestamp("2026-10-22 06:00", tz="UTC")


def test_json_is_serialisable_and_handles_missing_values():
    import json
    fc = fake_forecast(T0)
    fc.iloc[0, fc.columns.get_loc("national_grid")] = np.nan
    body = forecast_to_json(fc, T0, False)
    json.dumps(body)
    assert body["points"][0]["national_grid"] is None
    assert len(body["points"]) == 48

    best = best_to_json(best_window(fc, 2), T0, False)
    json.dumps(best)
    assert best["hours"] == 2
