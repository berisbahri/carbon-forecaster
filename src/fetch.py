"""
Phase 1: download raw data for the carbon intensity forecaster.

Pulls:
  - National half-hourly carbon intensity (National Grid forecast + actual)
    from the Carbon Intensity API (api.carbonintensity.org.uk)
  - Hourly historical weather from the Open-Meteo archive API for a few
    representative GB locations, averaged into one national series

Usage (from the project root, with the venv active):
    python -m src.fetch              # last 730 days
    python -m src.fetch --days 365   # last year only

Outputs (in data/):
    intensity.csv               time, forecast, actual   (half-hourly, UTC)
    weather_by_location.csv     raw hourly weather per location
    weather.csv                 hourly national average weather
"""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CI_BASE = "https://api.carbonintensity.org.uk"
CI_CHUNK_DAYS = 14  # the API caps the date range per request; 14 days is safe

OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
# The archive lags real time by a few days; the most recent days get filled
# later from Open-Meteo's forecast API when we make live predictions.
OM_ARCHIVE_LAG_DAYS = 5

WEATHER_VARS = [
    "wind_speed_100m",      # drives wind generation (hub height)
    "shortwave_radiation",  # drives solar generation
    "temperature_2m",       # drives demand
    "cloud_cover",
]

# Representative points: windy north, offshore wind, demand centres, solar south.
LOCATIONS = {
    "north_scotland": (57.5, -4.2),
    "north_sea_offshore": (54.5, 1.5),
    "north_england": (54.0, -2.0),
    "midlands": (52.5, -1.5),
    "south_england": (51.2, -1.0),
    "wales": (52.2, -3.6),
}

session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "User-Agent": "carbon-forecaster (student portfolio project)",
})


def get_json(url: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET a URL and return JSON, retrying with backoff on failure."""
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  request failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_intensity(start: datetime, end: datetime) -> pd.DataFrame:
    """National half-hourly carbon intensity between start and end (UTC)."""
    rows = []
    odd_records = 0
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CI_CHUNK_DAYS), end)
        url = (f"{CI_BASE}/intensity/"
               f"{chunk_start:%Y-%m-%dT%H:%MZ}/{chunk_end:%Y-%m-%dT%H:%MZ}")
        print(f"intensity {chunk_start:%Y-%m-%d} -> {chunk_end:%Y-%m-%d}")
        for item in get_json(url).get("data", []):
            values = item.get("intensity") or {}
            if "forecast" not in values or "actual" not in values:
                odd_records += 1
                if odd_records == 1:  # show the first one so we can see the shape
                    print(f"  note: record with missing fields: {item}")
            rows.append({
                "time": item.get("from"),
                "forecast": values.get("forecast"),
                "actual": values.get("actual"),
            })
        chunk_start = chunk_end
        time.sleep(0.5)  # be polite to a free API

    if odd_records:
        print(f"  {odd_records} records had missing fields (kept as blanks)")

    df = pd.DataFrame(rows).dropna(subset=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    # chunks share boundaries, so drop the overlapping rows
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def fetch_weather(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly weather per location, plus the national average."""
    frames = []
    for name, (lat, lon) in LOCATIONS.items():
        print(f"weather {name} {start} -> {end}")
        js = get_json(OM_ARCHIVE, params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "UTC",
        })
        df = pd.DataFrame(js["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["location"] = name
        frames.append(df)
        time.sleep(0.5)

    by_location = pd.concat(frames, ignore_index=True)
    national = (by_location.groupby("time")[WEATHER_VARS].mean()
                .dropna(how="all").reset_index())
    return by_location, national


def summarise(name: str, df: pd.DataFrame, expected_rows: int | None = None) -> None:
    print(f"\n{name}: {len(df):,} rows, {df['time'].min()} -> {df['time'].max()}")
    if expected_rows:
        print(f"  expected about {expected_rows:,} rows")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("  missing values:\n" + missing.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw data.")
    parser.add_argument("--days", type=int, default=730,
                        help="how many days of history to fetch (default 730)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=args.days)

    intensity = fetch_intensity(start, now)
    intensity.to_csv(DATA_DIR / "intensity.csv", index=False)
    summarise("intensity", intensity, expected_rows=args.days * 48)

    weather_end = now.date() - timedelta(days=OM_ARCHIVE_LAG_DAYS)
    by_location, national = fetch_weather(start.date(), weather_end)
    by_location.to_csv(DATA_DIR / "weather_by_location.csv", index=False)
    national.to_csv(DATA_DIR / "weather.csv", index=False)
    summarise("weather (national average)", national)

    print(f"\nSaved to {DATA_DIR}")


if __name__ == "__main__":
    main()