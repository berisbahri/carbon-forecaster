import numpy as np
import pandas as pd
import pytest

from src.best_time import best_window


def make_forecast(values) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(values), freq="30min", tz="UTC")
    return pd.DataFrame({"pred": np.asarray(values, dtype=float)}, index=idx)


def test_finds_the_dip():
    values = [200] * 48
    values[20:24] = [50, 50, 50, 50]  # a clean 2-hour patch
    fc = make_forecast(values)
    best = best_window(fc, hours=2)
    assert best["start"] == fc.index[20]
    assert best["avg_intensity"] == 50


def test_window_has_the_requested_length():
    best = best_window(make_forecast(range(48, 0, -1)), hours=3)
    assert best["end"] - best["start"] == pd.Timedelta(hours=3)


def test_rounds_up_to_whole_half_hours():
    assert best_window(make_forecast([100] * 10), hours=1.2)["hours"] == 1.5


def test_saving_compared_with_starting_now():
    best = best_window(make_forecast([200, 200, 100, 100]), hours=1)
    assert best["start_now_avg"] == 200
    assert best["saving_pct"] == pytest.approx(50)


def test_respects_finish_by_deadline():
    values = [200] * 48
    values[40:44] = [10] * 4  # cleanest patch, but too late
    fc = make_forecast(values)
    best = best_window(fc, hours=2, finish_by=fc.index[30])
    assert best["end"] <= fc.index[30]


def test_task_longer_than_forecast_raises():
    with pytest.raises(ValueError):
        best_window(make_forecast([100] * 4), hours=5)


def test_gaps_raise():
    with pytest.raises(ValueError):
        best_window(make_forecast([100, np.nan, 100, 100]), hours=1)
