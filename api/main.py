"""
Phase 5: the public API.

Run locally (from the project root, with the venv active):
    uvicorn api.main:app --reload

Then open:
    http://127.0.0.1:8000/docs                 interactive API docs
    http://127.0.0.1:8000/forecast             next 24h forecast
    http://127.0.0.1:8000/best-time?hours=2    cleanest 2-hour window
    http://127.0.0.1:8000/metrics              model vs baselines vs National Grid
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.service import (ForecastService, ForecastUnavailable, best_to_json,
                         forecast_to_json, parse_deadline)
from src.best_time import best_window
from src.evaluate import DATA_DIR
from src.predict import MODEL_PATH

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="UK Grid Carbon Intensity Forecaster",
    version="0.1.0",
    description="24-hour forecasts of GB electricity carbon intensity, "
                "and the lowest-carbon time to run a task.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"],
                   allow_headers=["*"])

service = ForecastService()


def _get_forecast():
    try:
        return service.get()
    except ForecastUnavailable as e:
        raise HTTPException(status_code=503, detail=f"Forecast unavailable: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.get("/forecast")
def forecast(region: str = Query("national", description="Only 'national' for now")):
    if region.lower() != "national":
        raise HTTPException(status_code=400,
                            detail="Only region=national is supported so far.")
    fc, issued_at, stale = _get_forecast()
    return forecast_to_json(fc, issued_at, stale)


@app.get("/best-time")
def best_time(
    hours: float = Query(2, gt=0, le=24, description="Task length in hours"),
    finish_by: datetime | None = Query(
        None, description="Optional deadline, e.g. 2026-10-22T07:00 (UK time if no timezone)"),
):
    fc, issued_at, stale = _get_forecast()
    try:
        best = best_window(fc, hours, finish_by=parse_deadline(finish_by))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return best_to_json(best, issued_at, stale)


@app.get("/metrics")
def metrics():
    path = DATA_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail="No metrics yet: run python -m src.train")
    return json.loads(path.read_text())


# Frontend (Phase 6): serve web/index.html at the root URL
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")
