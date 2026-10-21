"""
Phase 3: train LightGBM and score it with the Phase 2 harness.

Trains two models so you can see what the weather adds:
  - model             lags + calendar + weather
  - model_no_weather  lags + calendar only

Both are scored on the same test rows as the baselines and National Grid.

Usage (from the project root, with the venv active):
    python -m src.train

Outputs:
    models/model.joblib         the trained model + its feature list
    data/metrics.json           results table (replaces the Phase 2 version)
    data/test_predictions.csv   time, actual, every method's prediction

Oracle-weather caveat: training and testing use *observed* historical
weather, but a live forecast only has weather *forecasts*, which are less
accurate. Test scores for `model` are therefore somewhat optimistic; the
`model_no_weather` row is a lower bound. Say this in your README.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from src.evaluate import DATA_DIR, add_baselines, evaluate, load_intensity, time_split
from src.features import (FEATURES, FEATURES_NO_WEATHER, WEATHER_COLS,
                          build_features, load_weather)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VALIDATION_DAYS = 30  # tail of the training period, used for early stopping

PARAMS = dict(
    objective="l1",        # optimise absolute error, matching our MAE metric
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)


def fit(train: pd.DataFrame, features: list[str]) -> lgb.LGBMRegressor:
    """Pick the number of trees on a time-based validation slice, then
    refit on the whole training period with that number."""
    val_start = train.index.max() - pd.Timedelta(days=VALIDATION_DAYS)
    fit_part, val_part = train[train.index <= val_start], train[train.index > val_start]

    probe = lgb.LGBMRegressor(n_estimators=3000, **PARAMS)
    probe.fit(fit_part[features], fit_part["actual"],
              eval_set=[(val_part[features], val_part["actual"])],
              eval_metric="l1",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    best_n = probe.best_iteration_ or 500
    print(f"  best number of trees: {best_n}")

    model = lgb.LGBMRegressor(n_estimators=best_n, **PARAMS)
    model.fit(train[features], train["actual"])
    return model


def main() -> None:
    intensity = load_intensity()
    weather = load_weather()
    data = build_features(intensity, weather)
    data["forecast"] = intensity["forecast"]
    data = add_baselines(data)

    # keep rows with a target, a full week of history and weather;
    # the newest few days have no archived weather yet, so they drop out
    data = data.dropna(subset=["actual", "lag_168h", *WEATHER_COLS])
    train, test, cutoff = time_split(data)
    print(f"Train: {train.index.min()} -> {cutoff}  ({len(train):,} rows)")
    print(f"Test:  {test.index.min()} -> {test.index.max()}  ({len(test):,} rows)\n")

    print("Training model (with weather)")
    model = fit(train, FEATURES)
    print("Training model_no_weather")
    model_nw = fit(train, FEATURES_NO_WEATHER)

    test = test.copy()
    test["pred_model"] = model.predict(test[FEATURES])
    test["pred_model_no_weather"] = model_nw.predict(test[FEATURES_NO_WEATHER])

    pred_cols = ["pred_model", "pred_model_no_weather", "pred_yesterday",
                 "pred_last_week", "pred_national_grid"]
    table = evaluate(test, pred_cols)
    print("\nTest-set results (gCO2/kWh):")
    print(table.round(2).to_string())

    importance = (pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
                  .sort_values(ascending=False))
    print("\nFeature importance (share of total gain):")
    print((100 * importance / importance.sum()).round(1).head(10).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODELS_DIR / "model.joblib")

    test[["actual", *pred_cols]].to_csv(DATA_DIR / "test_predictions.csv")
    out = {
        "cutoff": cutoff.isoformat(),
        "test_start": test.index.min().isoformat(),
        "test_end": test.index.max().isoformat(),
        "results": table.round(3).to_dict(orient="index"),
        "notes": ["Weather features use observed weather (optimistic for 'model').",
                  "National Grid's stored forecast is short-range, not 24h ahead."],
    }
    (DATA_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {MODELS_DIR / 'model.joblib'} and updated {DATA_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()