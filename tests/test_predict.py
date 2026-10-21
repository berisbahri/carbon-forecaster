import numpy as np
import pandas as pd

from src.features import FEATURES, WEATHER_COLS
from src.predict import assemble


class EchoLag24:
    """Stand-in model: predicts the value from 24 hours earlier."""
    def predict(self, X):
        return X["lag_24h"].to_numpy()


def make_inputs(now: pd.Timestamp):
    hist_idx = pd.date_range(now.floor("30min") - pd.Timedelta(days=9),
                             now.floor("30min"), freq="30min")
    history = pd.DataFrame({"time": hist_idx,
                            "actual": np.arange(len(hist_idx), dtype=float)})
    w_idx = pd.date_range(now.floor("h"), periods=3 * 48, freq="30min")
    weather = pd.DataFrame(1.0, index=w_idx, columns=WEATHER_COLS)
    return history, weather


def test_predicts_the_next_24_hours():
    now = pd.Timestamp("2026-10-21 14:10", tz="UTC")
    history, weather = make_inputs(now)
    out = assemble(history, weather, {"model": EchoLag24(), "features": FEATURES}, now)

    assert len(out) == 48
    assert out.index[0] == pd.Timestamp("2026-10-21 14:30", tz="UTC")
    assert out["pred"].notna().all()


def test_uses_actuals_from_24_hours_before_each_target():
    now = pd.Timestamp("2026-10-21 14:10", tz="UTC")
    history, weather = make_inputs(now)
    out = assemble(history, weather, {"model": EchoLag24(), "features": FEATURES}, now)

    lookup = history.set_index("time")["actual"]
    t = out.index[10]
    assert out.loc[t, "pred"] == lookup[t - pd.Timedelta(hours=24)]
