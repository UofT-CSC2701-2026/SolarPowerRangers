import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import storage
from pydantic import BaseModel, Field, model_validator

from hourly_forecast import ForecastUnavailable, get_grid_forecast, load_hourly_model

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("solar-api")

GCS_BUCKET = os.getenv("GCS_BUCKET")
WEB_DIRECTORY = Path(__file__).resolve().parent / "web"
MODEL_BLOB = os.getenv(
    "MODEL_BLOB",
    "models/solar_xgboost/solar_xgboost_model.json"
)
METADATA_BLOB = os.getenv(
    "METADATA_BLOB",
    "models/solar_xgboost/metadata.json"
)

DAYLIGHT_THRESHOLD = 20.0
INTERVAL_HOURS = 1.0

FEATURE_COLUMNS = [
    "daily_irradiation_kwh_m2",
    "max_irradiance_wm2",
    "daylight_hours",
    "ambient_temp_mean",
    "ambient_temp_min",
    "ambient_temp_max",
    "cloud_cover_mean",
    "cloud_cover_max",
    "day_of_week",
    "day_of_year",
    "month",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
]

state = {"model": None, "metadata": {}}


def load_model_from_gcs():
    """Download the versioned XGBoost JSON artifact once at startup."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    model_blob = bucket.blob(MODEL_BLOB)
    if not model_blob.exists():
        raise RuntimeError(
            f"Model artifact not found: gs://{GCS_BUCKET}/{MODEL_BLOB}"
        )

    model_bytes = model_blob.download_as_bytes()
    model = xgb.Booster()
    
    # XGBoost's sklearn wrapper loads its native JSON model from a file.
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        tmp.write(model_bytes)
        tmp.flush()
        model.load_model(tmp.name)

    metadata = {}
    metadata_blob = bucket.blob(METADATA_BLOB)
    if metadata_blob.exists():
        metadata = json.loads(metadata_blob.download_as_text())

    # Fail fast if serving features do not match the training contract.
    trained_features = metadata.get("features")
    if trained_features and trained_features != FEATURE_COLUMNS:
        raise RuntimeError(
            "Feature mismatch between API and model metadata. "
            f"Expected {FEATURE_COLUMNS}; found {trained_features}"
        )

    state["model"] = model
    state["metadata"] = metadata
    logger.info("Loaded model from gs://%s/%s", GCS_BUCKET, MODEL_BLOB)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if GCS_BUCKET:
        load_model_from_gcs()
    else:
        load_hourly_model()
    yield
    state["model"] = None


app = FastAPI(
    title="Solar Daily Energy Prediction API",
    version="1.0.0",
    description=(
        "Predict daily solar energy generation in kWh from "
        "24 hourly weather forecast observations."
    ),
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=WEB_DIRECTORY), name="forecast-assets")


@app.get("/", include_in_schema=False)
def forecast_page():
    return FileResponse(WEB_DIRECTORY / "index.html")


@app.get("/forecast/canada")
def canada_forecast():
    try:
        return get_grid_forecast()
    except ForecastUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


class HourlyForecast(BaseModel):
    hour: int = Field(ge=0, le=23, description="Local plant hour, 0 through 23")
    solar_irradiance_wm2: float = Field(ge=0)
    ambient_temperature_c: float
    cloud_cover_percent: float = Field(ge=0, le=100)


class DailyPredictionRequest(BaseModel):
    forecast_date: date
    hourly_forecasts: List[HourlyForecast] = Field(
        min_length=24,
        max_length=24
    )

    @model_validator(mode="after")
    def validate_hours(self):
        hours = [item.hour for item in self.hourly_forecasts]
        if len(set(hours)) != 24 or set(hours) != set(range(24)):
            raise ValueError(
                "hourly_forecasts must contain exactly one observation "
                "for every hour from 0 through 23"
            )
        return self


def build_daily_features(request: DailyPredictionRequest) -> pd.DataFrame:
    """Mirror the daily feature transformations in the training DAG."""
    hourly = pd.DataFrame(
        [item.model_dump() for item in request.hourly_forecasts]
    ).sort_values("hour")

    irradiance = hourly["solar_irradiance_wm2"]
    ambient = hourly["ambient_temperature_c"]
    cloud = hourly["cloud_cover_percent"]

    day_of_week = request.forecast_date.weekday()
    day_of_year = request.forecast_date.timetuple().tm_yday
    month = request.forecast_date.month

    features = {
        "daily_irradiation_kwh_m2": (
            irradiance.sum() * INTERVAL_HOURS / 1000.0
        ),
        "max_irradiance_wm2": irradiance.max(),
        "daylight_hours": int((irradiance > DAYLIGHT_THRESHOLD).sum()),
        "ambient_temp_mean": ambient.mean(),
        "ambient_temp_min": ambient.min(),
        "ambient_temp_max": ambient.max(),
        "cloud_cover_mean": cloud.mean(),
        "cloud_cover_max": cloud.max(),
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "month": month,
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
    }

    # Preserve the exact training feature order.
    return pd.DataFrame(
        [[features[name] for name in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )


@app.get("/health")
def health():
    if GCS_BUCKET and state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.get("/model/info")
def model_info():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "model_type": "XGBoost Regressor",
        "target": "daily_energy_kwh",
        "prediction_unit": "kWh/day",
        "model_blob": f"gs://{GCS_BUCKET}/{MODEL_BLOB}",
        "metadata": state["metadata"],
    }


@app.post("/predict/daily")
def predict_daily(request: DailyPredictionRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        X = build_daily_features(request)
        dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
        prediction = float(state["model"].predict(dmatrix)[0])

        # Physical lower bound: solar energy cannot be negative.
        prediction = max(0.0, prediction)

        return {
            "forecast_date": request.forecast_date.isoformat(),
            "predicted_daily_energy_kwh": round(prediction, 3),
            "unit": "kWh",
            "model": "solar_xgboost",
            "feature_values": X.iloc[0].to_dict(),
        }
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc
