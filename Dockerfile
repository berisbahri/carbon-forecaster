# Match this to the Python version you trained with (python --version)
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY web/ web/
COPY models/model.joblib models/model.joblib
COPY data/metrics.json data/metrics.json

# Cloud Run tells the container which port to use via $PORT
ENV PORT=8080
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
