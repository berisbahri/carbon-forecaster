"""
Phase 4: live 24-hour forecast.

  1. Fetch the last 9 days of actual intensity (enough for the 7-day lags)
  2. Fetch Open-Meteo's weather forecast for the same locations as training
  3. Fetch National Grid's own forecast for the next 24 hours, to compare
  4. Build features with exactly the same code as training, then predict

Usage (from the project root, with the venv active):
    python -m src.predict             # also shows the best 2-hour window
    python -m src.predict --hours 4   # best 4-hour window

Outputs:
    data/latest_forecast.csv   time, pred, national_grid (next 24h)
    data/forecast_log.csv      every forecast ever made, with National
                               Grid's forecast as it stood at that moment.
                               After a few weeks this gives a fair,
                               like-for-like 24h-ahead comparison.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd

from src.best_time import best_window
from src.evaluate import DATA_DIR
from src.features import HORIZON_SLOTS, WEATHER_COLS, build_features
from src.fetch import CI_BASE, LOCATIONS, fetch_intensity, get_json

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
OM_FORECAST = "https://api.open-meteo.com/v1/forecast"
HISTORY_DAYS = 9
SLOT = pd.Timedelta(minutes=30)


def fetch_history(now: pd.Timestamp) -> pd.DataFrame:
    return fetch_intensity(now - timedelta(days=HISTORY_DAYS), now)


def fetch_national_grid_forecast(now: pd.Timestamp) -> pd.Series:
    url = f"{CI_BASE}/intensity/{now:%Y-%m-%dT%H:%MZ}/fw24h"
    rows = [{"time": item.get("from"),
             "national_grid": (item.get("intensity") or {}).get("forecast")}
            for item in get_json(url).get("data", [])]
    df = pd.DataFrame(rows).dropna(subset=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop_duplicates("time").set_index("time")["national_grid"]


def fetch_weather_forecast() -> pd.DataFrame:
    """Hourly forecast averaged over the training locations, on a 30-min grid."""
    frames = []
    for lat, lon in LOCATIONS.values():
        js = get_json(OM_FORECAST, params={
            "latitude": lat, "longitude": lon,
            "hourly": ",".join(WEATHER_COLS),
            "forecast_days": 3, "timezone": "UTC",
        })
        df = pd.DataFrame(js["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        frames.append(df)
    weather = pd.concat(frames).groupby("time")[WEATHER_COLS].mean()
    return weather.resample("30min").interpolate(limit=1)


def assemble(history: pd.DataFrame, weather: pd.DataFrame, bundle: dict,
             now: pd.Timestamp, national_grid: pd.Series | None = None) -> pd.DataFrame:
    """Predict the 48 half-hours after `now`. Pure function, easy to test.

    history  columns time, actual (as returned by fetch_intensity)
    bundle   {"model": ..., "features": [...]} from models/model.joblib
    """
    first = now.floor("30min") + SLOT
    targets = pd.date_range(first, periods=HORIZON_SLOTS, freq="30min")

    h = history.drop_duplicates("time").set_index("time").sort_index()
    grid = pd.date_range(h.index.min(), targets[-1], freq="30min")
    intensity = h[["actual"]].reindex(grid)  # future rows are blank

    features = build_features(intensity, weather)
    X = features.loc[targets, bundle["features"]]

    out = pd.DataFrame({"pred": bundle["model"].predict(X)}, index=targets)
    out.index.name = "time"
    out["national_grid"] = (national_grid.reindex(targets)
                            if national_grid is not None else float("nan"))
    return out


def make_forecast(now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Fetch everything and return the next 24 hours of predictions."""
    now = now or pd.Timestamp.now(tz="UTC")
    bundle = joblib.load(MODEL_PATH)
    history = fetch_history(now.floor("30min"))
    weather = fetch_weather_forecast()
    try:
        ng = fetch_national_grid_forecast(now.floor("30min"))
    except Exception as e:  # our forecast still works without it
        print(f"warning: could not fetch National Grid forecast ({e})")
        ng = None
    return assemble(history, weather, bundle, now, ng)


def save(forecast: pd.DataFrame, issued_at: pd.Timestamp) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    forecast.to_csv(DATA_DIR / "latest_forecast.csv")
    log_path = DATA_DIR / "forecast_log.csv"
    forecast.assign(issued_at=issued_at.isoformat()).to_csv(
        log_path, mode="a", header=not log_path.exists())


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a live 24h forecast.")
    parser.add_argument("--hours", type=float, default=2,
                        help="task length for the best-time recommendation")
    args = parser.parse_args()

    now = pd.Timestamp.now(tz="UTC")
    forecast = make_forecast(now)
    save(forecast, now)

    local = forecast.tz_convert("Europe/London")
    print("Next 24 hours (UK time, gCO2/kWh), every 2 hours:")
    print(local.iloc[::4].round(0).to_string())

    best = best_window(forecast, args.hours)
    fmt = "%a %H:%M"
    print(f"\nBest {best['hours']:g}-hour window: "
          f"{best['start'].tz_convert('Europe/London'):{fmt}} to "
          f"{best['end'].tz_convert('Europe/London'):{fmt}}, "
          f"average {best['avg_intensity']:.0f} gCO2/kWh "
          f"({best['saving_pct']:.0f}% lower than starting now)")
    print(f"\nSaved {DATA_DIR / 'latest_forecast.csv'}")


if __name__ == "__main__":
    main()
