"""Serve the rolling Canada forecast with the repository's hourly model."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from itertools import repeat
from pathlib import Path
from threading import Lock
from time import monotonic
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_FIELDS = ("shortwave_radiation", "temperature_2m", "cloud_cover")
FEATURES = [
    "solar_irradiance_wm2", "ambient_temperature_c", "panel_temperature_c",
    "cloud_cover_percent", "hour", "month", "dayofyear", "dayofweek",
]
BATCH_SIZE = 20
HOUR = timedelta(hours=1)
TRAINING_TEMP_RANGE_C = (9.12, 49.3)
logger = logging.getLogger(__name__)
cache_lock = Lock()
cached_result = None
cache_expiry = 0
retry_after = 0


class ForecastUnavailable(Exception):
    """The weather forecast could not be retrieved."""


@lru_cache(maxsize=1)
def load_cells():
    return json.loads((ROOT / "web" / "canada-grid.json").read_text())["cells"]


@lru_cache(maxsize=1)
def load_hourly_model():
    model = xgb.Booster(params={"nthread": 1})
    model.load_model(ROOT / "solar_xgboost_model.json")
    if model.feature_names != FEATURES:
        raise RuntimeError("The hourly model's input columns do not match the forecast service")
    return model


def fetch_weather(cells, window_start):
    # Open-Meteo's irradiance is averaged over the hour ending at the timestamp.
    start = window_start + HOUR
    end = window_start + 24 * HOUR
    params = {
        "latitude": ",".join(str(cell["latitude"]) for cell in cells),
        "longitude": ",".join(str(cell["longitude"]) for cell in cells),
        "hourly": ",".join(WEATHER_FIELDS),
        "timezone": "GMT",
        "timeformat": "unixtime",
        "start_hour": start.strftime("%Y-%m-%dT%H:%M"),
        "end_hour": end.strftime("%Y-%m-%dT%H:%M"),
    }
    try:
        with urlopen(f"{FORECAST_URL}?{urlencode(params)}", timeout=12) as response:
            payload = json.load(response)
    except (OSError, ValueError) as exc:
        logger.warning("Open-Meteo batch of %s grid cells failed: %s", len(cells), exc)
        return []

    forecasts = payload if isinstance(payload, list) else [payload]
    if len(forecasts) != len(cells):
        logger.warning("Open-Meteo returned %s locations for %s cells", len(forecasts), len(cells))
        return []
    return list(zip(cells, forecasts))


def build_hourly_features(cell, weather, timestamps):
    hourly = weather.get("hourly", {})
    if hourly.get("time") != timestamps:
        return None
    series = [hourly.get(field, []) for field in WEATHER_FIELDS]
    if any(len(values) != len(timestamps) for values in series):
        return None

    zone = ZoneInfo(cell["timezone"])
    features = []
    for timestamp, (irradiance, ambient, cloud) in zip(timestamps, zip(*series)):
        if None in (irradiance, ambient, cloud):
            return None
        local_time = datetime.fromtimestamp(timestamp, zone)
        # Linear panel-temperature fit from model.ipynb.
        panel_temp = -0.000340368 + 1.00001288 * ambient + 0.012500128 * irradiance
        features.append([
            irradiance, ambient, panel_temp, cloud,
            local_time.hour, local_time.month, local_time.timetuple().tm_yday, local_time.weekday(),
        ])
    return features


def build_grid_forecast(now):
    start = now.replace(minute=0, second=0, microsecond=0)
    if now > start:
        start += HOUR
    timestamps = [int((start + i * HOUR).timestamp()) for i in range(1, 25)]
    cells = load_cells()
    batches = [cells[i:i + BATCH_SIZE] for i in range(0, len(cells), BATCH_SIZE)]
    available = []
    rows = []
    low, high = TRAINING_TEMP_RANGE_C
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in pool.map(fetch_weather, batches, repeat(start)):
            for cell, weather in batch:
                features = build_hourly_features(cell, weather, timestamps)
                if features is not None:
                    outside_range = any(
                        irradiance > 20 and not low <= ambient <= high
                        for irradiance, ambient, *_ in features
                    )
                    available.append((cell["id"], outside_range))
                    rows.extend(features)

    if not available:
        raise ForecastUnavailable("Weather forecasts are temporarily unavailable. Try again shortly.")

    matrix = xgb.DMatrix(np.asarray(rows, dtype=np.float32), feature_names=FEATURES, nthread=1)
    predictions = load_hourly_model().predict(matrix).clip(min=0).reshape(len(available), 24)
    forecasts = {}
    for (cell_id, outside_range), power in zip(available, predictions):
        hourly_kw = power.astype(float).round(3).tolist()
        forecasts[cell_id] = {
            "hourly_kw": hourly_kw,
            "energy_kwh": round(sum(hourly_kw), 3),
            "daylight_temperature_outside_training_range": outside_range,
        }

    return {
        "updated_at": now.isoformat(),
        "window_start": start.isoformat(),
        "hour_ends": timestamps,
        "cells": forecasts,
        "highest_cell": max(forecasts, key=lambda cell_id: forecasts[cell_id]["energy_kwh"]),
    }


def get_grid_forecast():
    global cached_result, cache_expiry, retry_after
    with cache_lock:
        if cached_result is not None and monotonic() < cache_expiry:
            return cached_result
        if monotonic() < retry_after:
            return {**cached_result, "stale": True}
        try:
            result = build_grid_forecast(datetime.now(timezone.utc))
        except ForecastUnavailable:
            if cached_result is None or monotonic() >= cache_expiry + 3600:
                raise
            retry_after = min(monotonic() + 60, cache_expiry + 3600)
            return {**cached_result, "stale": True}

        complete = len(result["cells"]) == len(load_cells())
        cache_expiry = monotonic() + (1800 if complete else 300)
        retry_after = 0
        cached_result = result
        return cached_result
