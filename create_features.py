
from datetime import datetime, timedelta
from io import BytesIO
import json
import logging

import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# ==========================================================
# CONFIGURATION
# ==========================================================

GCP_CONN_ID = "google_cloud_default"

BUCKET_NAME = "solar-power-storage"

INPUT_FILE = "data/raw/solar_plant_generation_dataset.csv"

OUTPUT_PREFIX = "data/processed/daily_solar_features"

TIMESTAMP = "timestamp"

TARGET = "daily_energy_kwh"

# Assumption: each power reading represents the average
# power over the preceding/following one-hour interval.
INTERVAL_HOURS = 1.0

# Irradiance threshold used to identify daylight
DAYLIGHT_THRESHOLD = 20.0

EXPECTED_COLUMNS = [
    "timestamp",
    "solar_irradiance_wm2",
    "ambient_temperature_c",
    "panel_temperature_c",
    "cloud_cover_percent",
    "power_generation_kw"
]

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
    "month_cos"
]


# ==========================================================
# 1. READ DATA FROM GCS
# ==========================================================

def read_from_gcs():

    gcs = GCSHook(gcp_conn_id=GCP_CONN_ID)

    csv_data = gcs.download(
        bucket_name=BUCKET_NAME,
        object_name=INPUT_FILE
    )

    df = pd.read_csv(BytesIO(csv_data))

    logging.info("Dataset shape: %s", df.shape)

    return df


# ==========================================================
# 2. DATA VALIDATION
# ==========================================================

def validate_data(df):

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected schema. Expected: {EXPECTED_COLUMNS}"
        )

    if df.isnull().any().any():
        raise ValueError("Missing values found in dataset")

    df = df.copy()

    df[TIMESTAMP] = pd.to_datetime(
        df[TIMESTAMP],
        errors="raise"
    )

    numeric_columns = [
        col for col in EXPECTED_COLUMNS
        if col != TIMESTAMP
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

    if not df[TIMESTAMP].is_monotonic_increasing:
        raise ValueError("Timestamps are not chronological")

    if df[TIMESTAMP].duplicated().any():
        raise ValueError("Duplicate timestamps detected")

    # Verify hourly frequency
    differences = df[TIMESTAMP].diff().dropna()

    if not (differences == pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Dataset contains missing or irregular hourly intervals"
        )

    if (df["power_generation_kw"] < 0).any():
        raise ValueError("Negative power generation detected")

    if (df["solar_irradiance_wm2"] < 0).any():
        raise ValueError("Negative solar irradiance detected")

    if not df["cloud_cover_percent"].between(0, 100).all():
        raise ValueError("Invalid cloud cover values")

    return df


# ==========================================================
# 3. DAILY AGGREGATION
# ==========================================================

def aggregate_daily_data(df):

    df = df.copy()

    df["date"] = df[TIMESTAMP].dt.normalize()

    # Keep only complete 24-hour days.
    daily_counts = df.groupby("date").size()

    complete_dates = daily_counts[
        daily_counts == 24
    ].index

    df = df[df["date"].isin(complete_dates)].copy()

    if df.empty:
        raise ValueError("No complete days found")

    # Verify that each complete day has all 24 hours.
    valid_hours = (
        df.groupby("date")[TIMESTAMP]
        .apply(lambda x: set(x.dt.hour) == set(range(24)))
    )

    if not valid_hours.all():
        raise ValueError("Incomplete hourly coverage detected")

    # Aggregate hourly weather and power observations.
    daily = df.groupby("date").agg(

        # Sum of hourly irradiance converted to kWh/m2
        daily_irradiation_kwh_m2=(
            "solar_irradiance_wm2",
            lambda x: x.sum() * INTERVAL_HOURS / 1000.0
        ),

        max_irradiance_wm2=(
            "solar_irradiance_wm2",
            "max"
        ),

        daylight_hours=(
            "solar_irradiance_wm2",
            lambda x: (x > DAYLIGHT_THRESHOLD).sum()
        ),

        ambient_temp_mean=(
            "ambient_temperature_c",
            "mean"
        ),

        ambient_temp_min=(
            "ambient_temperature_c",
            "min"
        ),

        ambient_temp_max=(
            "ambient_temperature_c",
            "max"
        ),

        cloud_cover_mean=(
            "cloud_cover_percent",
            "mean"
        ),

        cloud_cover_max=(
            "cloud_cover_percent",
            "max"
        ),

        # Daily energy in kWh
        daily_energy_kwh=(
            "power_generation_kw",
            lambda x: x.sum() * INTERVAL_HOURS
        )

    ).reset_index()

    return daily


# ==========================================================
# 4. FEATURE ENGINEERING
# ==========================================================

def create_features(daily):

    daily = daily.copy()

    daily["day_of_week"] = daily["date"].dt.dayofweek

    daily["day_of_year"] = (
        daily["date"].dt.dayofyear
    )

    daily["month"] = daily["date"].dt.month

    # Cyclical day-of-year features
    daily["day_of_year_sin"] = np.sin(
        2 * np.pi * daily["day_of_year"] / 365.25
    )

    daily["day_of_year_cos"] = np.cos(
        2 * np.pi * daily["day_of_year"] / 365.25
    )

    # Cyclical month features
    daily["month_sin"] = np.sin(
        2 * np.pi * (daily["month"] - 1) / 12
    )

    daily["month_cos"] = np.cos(
        2 * np.pi * (daily["month"] - 1) / 12
    )

    daily = daily.sort_values("date").reset_index(drop=True)

    return daily


# ==========================================================
# 5. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

def split_dataset(df):

    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)

    # For 208 complete days:
    # Train = 146
    # Validation = 31
    # Test = 31

    train_end = round(n * 0.70)

    validation_end = (
        train_end + round(n * 0.15)
    )

    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    logging.info(
        "Train: %s, Validation: %s, Test: %s",
        len(train_df),
        len(validation_df),
        len(test_df)
    )

    # Ensure chronological separation
    assert train_df["date"].max() < validation_df["date"].min()

    assert (
        validation_df["date"].max()
        < test_df["date"].min()
    )

    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df
    }


# ==========================================================
# 6. UPLOAD PARQUET FILES TO GCS
# ==========================================================

def upload_to_gcs(datasets, daily):

    gcs = GCSHook(gcp_conn_id=GCP_CONN_ID)

    for split_name, split_df in datasets.items():

        output_columns = [
            "date"
        ] + FEATURE_COLUMNS + [TARGET]

        output_df = split_df[output_columns]

        buffer = BytesIO()

        output_df.to_parquet(
            buffer,
            index=False,
            engine="pyarrow"
        )

        buffer.seek(0)

        output_path = (
            f"{OUTPUT_PREFIX}/{split_name}.parquet"
        )

        gcs.upload(
            bucket_name=BUCKET_NAME,
            object_name=output_path,
            data=buffer.getvalue(),
            mime_type="application/octet-stream"
        )

        logging.info(
            "Uploaded %s records to %s",
            len(output_df),
            output_path
        )

    # Upload feature manifest
    manifest = {
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "total_daily_records": len(daily),
        "train_records": len(datasets["train"]),
        "validation_records": len(datasets["validation"]),
        "test_records": len(datasets["test"]),
        "first_date": str(daily["date"].min().date()),
        "last_date": str(daily["date"].max().date()),
        "split_strategy": "chronological",
        "excluded_features": [
            "panel_temperature_c",
            "timestamp",
            "power_generation_kw"
        ],
        "target_definition": (
            "Sum of hourly power in kW multiplied by "
            "one-hour intervals, resulting in kWh"
        )
    }

    gcs.upload(
        bucket_name=BUCKET_NAME,
        object_name=f"{OUTPUT_PREFIX}/feature_manifest.json",
        data=json.dumps(manifest, indent=4),
        mime_type="application/json"
    )


# ==========================================================
# 7. MAIN PIPELINE
# ==========================================================

def prepare_daily_solar_dataset():

    logging.info("Starting solar dataset preparation")

    # Read
    df = read_from_gcs()

    # Validate
    df = validate_data(df)

    # Aggregate
    daily = aggregate_daily_data(df)

    logging.info(
        "Daily dataset shape after aggregation: %s",
        daily.shape
    )

    # Feature engineering
    daily = create_features(daily)

    # Validate engineered features
    if daily[FEATURE_COLUMNS + [TARGET]].isnull().any().any():
        raise ValueError("Null values found after feature engineering")

    # Split chronologically
    datasets = split_dataset(daily)

    # Upload
    upload_to_gcs(datasets, daily)

    logging.info("Solar dataset preparation completed")


# ==========================================================
# 8. AIRFLOW DAG DEFINITION
# ==========================================================

default_args = {
    "owner": "solar-ml-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5)
}


with DAG(

    dag_id="daily_solar_power_feature_engineering",

    default_args=default_args,

    description=(
        "Daily solar power dataset preparation "
        "for XGBoost regression"
    ),

    start_date=datetime(2025, 1, 1),

    schedule_interval=None,

    catchup=False,

    tags=[
        "solar",
        "xgboost",
        "daily-forecast",
        "feature-engineering"
    ]

) as dag:

    prepare_dataset = PythonOperator(

        task_id="prepare_daily_solar_dataset",

        python_callable=prepare_daily_solar_dataset

    )

    trigger_training = TriggerDagRunOperator(

        task_id="trigger_xgboost_training",

        trigger_dag_id="daily_solar_xgboost_training",

        wait_for_completion=False,

        poke_interval=30,

        allowed_states=["success"],

        failed_states=["failed"]

    )
    
    prepare_dataset >> trigger_training
