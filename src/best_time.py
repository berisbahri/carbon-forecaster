"""
Phase 4: find the lowest-carbon window to run a task.

Given a half-hourly forecast, slide a window of the task's length along it
and pick the window with the lowest average intensity.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SLOT = pd.Timedelta(minutes=30)


def best_window(forecast: pd.DataFrame, hours: float, col: str = "pred",
                finish_by: pd.Timestamp | None = None) -> dict:
    """Cleanest contiguous window of `hours` length in `forecast`.

    forecast   DataFrame indexed by half-hour start times (UTC), with `col`
    hours      task length; rounded up to whole half-hours (1.2h -> 1.5h)
    finish_by  optional deadline: the task must end by this time

    Returns start, end, the window's average intensity, and how that
    compares with starting straight away.
    """
    if hours <= 0:
        raise ValueError("hours must be positive")
    slots = math.ceil(hours * 2)

    series = forecast[col]
    if finish_by is not None:
        series = series[series.index + SLOT <= finish_by]

    values = series.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("forecast has gaps; cannot pick a window")
    if slots > len(values):
        raise ValueError(f"a {hours}h task needs {slots} half-hours but only "
                         f"{len(values)} are available")

    # average of every run of `slots` consecutive half-hours
    means = np.convolve(values, np.ones(slots) / slots, mode="valid")
    i = int(np.argmin(means))
    start_now = float(means[0])

    return {
        "start": series.index[i],
        "end": series.index[i + slots - 1] + SLOT,
        "hours": slots / 2,
        "avg_intensity": float(means[i]),
        "start_now_avg": start_now,
        "saving_pct": float(100 * (1 - means[i] / start_now)) if start_now > 0 else 0.0,
    }
