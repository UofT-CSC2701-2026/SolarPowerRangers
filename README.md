# SolarPowerRangers

[Kaggle Dataset](https://www.kaggle.com/datasets/juanschafle/solarplant-power-generation-monitoring-2023-2025)

# Solar Daily Energy Prediction API

FastAPI service for daily solar energy prediction using the XGBoost model stored in GCS.

## Contract

`POST /predict/daily` accepts exactly 24 hourly forecast records, one for each local plant hour 0–23.
Required forecast fields:
- `solar_irradiance_wm2`
- `ambient_temperature_c`
- `cloud_cover_percent`

The API creates the same 15 daily features as the training pipeline and returns predicted `kWh/day`.

## Environment variables

- `GCS_BUCKET`: bucket name only, without `gs://`
- `MODEL_BLOB`: default `models/solar_xgboost/solar_xgboost_model.json`
- `METADATA_BLOB`: default `models/solar_xgboost/metadata.json`

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GCS_BUCKET=solar-power-storage
gcloud auth application-default login
uvicorn main:app --reload --port 8080
```

The service account/user needs `roles/storage.objectViewer` on the model bucket.

## Example request

```json
{
  "forecast_date": "2026-11-16",
  "hourly_forecasts": [
    {"hour": 0, "solar_irradiance_wm2": 0, "ambient_temperature_c": 8, "cloud_cover_percent": 80},
    {"hour": 1, "solar_irradiance_wm2": 0, "ambient_temperature_c": 8, "cloud_cover_percent": 80},
    {"hour": 2, "solar_irradiance_wm2": 0, "ambient_temperature_c": 7, "cloud_cover_percent": 80},
    {"hour": 3, "solar_irradiance_wm2": 0, "ambient_temperature_c": 7, "cloud_cover_percent": 80},
    {"hour": 4, "solar_irradiance_wm2": 0, "ambient_temperature_c": 7, "cloud_cover_percent": 80},
    {"hour": 5, "solar_irradiance_wm2": 0, "ambient_temperature_c": 7, "cloud_cover_percent": 80},
    {"hour": 6, "solar_irradiance_wm2": 0, "ambient_temperature_c": 7, "cloud_cover_percent": 80},
    {"hour": 7, "solar_irradiance_wm2": 30, "ambient_temperature_c": 8, "cloud_cover_percent": 70},
    {"hour": 8, "solar_irradiance_wm2": 100, "ambient_temperature_c": 9, "cloud_cover_percent": 60},
    {"hour": 9, "solar_irradiance_wm2": 250, "ambient_temperature_c": 11, "cloud_cover_percent": 50},
    {"hour": 10, "solar_irradiance_wm2": 400, "ambient_temperature_c": 13, "cloud_cover_percent": 40},
    {"hour": 11, "solar_irradiance_wm2": 550, "ambient_temperature_c": 15, "cloud_cover_percent": 30},
    {"hour": 12, "solar_irradiance_wm2": 650, "ambient_temperature_c": 17, "cloud_cover_percent": 25},
    {"hour": 13, "solar_irradiance_wm2": 600, "ambient_temperature_c": 18, "cloud_cover_percent": 25},
    {"hour": 14, "solar_irradiance_wm2": 450, "ambient_temperature_c": 18, "cloud_cover_percent": 30},
    {"hour": 15, "solar_irradiance_wm2": 300, "ambient_temperature_c": 17, "cloud_cover_percent": 40},
    {"hour": 16, "solar_irradiance_wm2": 150, "ambient_temperature_c": 15, "cloud_cover_percent": 50},
    {"hour": 17, "solar_irradiance_wm2": 40, "ambient_temperature_c": 13, "cloud_cover_percent": 60},
    {"hour": 18, "solar_irradiance_wm2": 0, "ambient_temperature_c": 12, "cloud_cover_percent": 70},
    {"hour": 19, "solar_irradiance_wm2": 0, "ambient_temperature_c": 11, "cloud_cover_percent": 70},
    {"hour": 20, "solar_irradiance_wm2": 0, "ambient_temperature_c": 10, "cloud_cover_percent": 75},
    {"hour": 21, "solar_irradiance_wm2": 0, "ambient_temperature_c": 10, "cloud_cover_percent": 75},
    {"hour": 22, "solar_irradiance_wm2": 0, "ambient_temperature_c": 9, "cloud_cover_percent": 80},
    {"hour": 23, "solar_irradiance_wm2": 0, "ambient_temperature_c": 9, "cloud_cover_percent": 80}
  ]
}
```

## Deploy to Cloud Run

1. Enable Cloud Run, Cloud Build, and Artifact Registry APIs.
2. Give the Cloud Run runtime service account `roles/storage.objectViewer` on the model bucket.
3. Build and deploy from this directory:

```bash
gcloud run deploy solar-prediction-api \
  --source . \
  --region northamerica-northeast1 \
  --service-account solar-api-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars GCS_BUCKET=solar-power-storage,MODEL_BLOB=models/solar_xgboost/solar_xgboost_model.json,METADATA_BLOB=models/solar_xgboost/metadata.json \
  --memory 2Gi \
  --cpu 2 \
  --min 0 \
  --max 5 \
  --no-allow-unauthenticated
```


## Important model limitation

The current training data uses historical observed weather. A production prediction request uses forecast weather. The model can technically serve these inputs, but forecast-vs-observation differences can reduce accuracy. For a reliable production model, retrain with archived weather forecasts corresponding to each historical prediction date, using only forecasts available at the prediction cutoff time.

Also verify that each hourly power reading represents an hourly average before interpreting the training target as kWh/day.
