# Grid Carbon Intensity Forecaster

Forecasts UK grid carbon intensity for the next 24 hours and recommends the lowest-carbon time to run a task.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt

python -m src.fetch                 # download ~2 years of data
python -m src.train                 # train and evaluate the model
python -m uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000 for the app, or http://127.0.0.1:8000/docs for the API.

Run the tests with `python -m pytest`.