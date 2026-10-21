"""Endpoint tests using FastAPI's test client, with the network stubbed out."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as main
from api.service import ForecastService


def fake_forecast(now: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(now.floor("30min") + pd.Timedelta(minutes=30),
                        periods=48, freq="30min")
    values = np.full(48, 200.0)
    values[10:14] = 50.0
    return pd.DataFrame({"pred": values, "national_grid": values + 5}, index=idx)


@pytest.fixture
def client(monkeypatch):
    stub = ForecastService(forecaster=fake_forecast, on_refresh=lambda *a: None,
                           max_age=timedelta(minutes=30))
    monkeypatch.setattr(main, "service", stub)
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_forecast_returns_48_points(client):
    body = client.get("/forecast").json()
    assert len(body["points"]) == 48
    assert body["unit"] == "gCO2/kWh"


def test_unknown_region_is_rejected(client):
    assert client.get("/forecast?region=scotland").status_code == 400


def test_best_time_finds_the_clean_window(client):
    body = client.get("/best-time?hours=2").json()
    assert body["avg_intensity"] == 50.0


def test_best_time_validates_hours(client):
    assert client.get("/best-time?hours=0").status_code == 422
    assert client.get("/best-time?hours=100").status_code == 422
