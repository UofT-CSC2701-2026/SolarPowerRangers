
from datetime import datetime, timedelta
from io import BytesIO
import json
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook


# ==========================================================
# CONFIGURATION
# ==========================================================

GCP_CONN_ID = "google_cloud_default"

BUCKET_NAME = "solar-power-storage"

FEATURE_PREFIX = "data/processed/daily_solar_features"

MODEL_PREFIX = "models/solar_xgboost"

TARGET = "daily_energy_kwh"

RANDOM_STATE = 42

MAX_ESTIMATORS = 1000

EARLY_STOPPING_ROUNDS = 30


# Must match the feature engineering DAG
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
# 1. LOAD DATA FROM GCS
# ==========================================================

def load_parquet(gcs, filename):

    path = f"{FEATURE_PREFIX}/{filename}"

    logging.info("Reading GCS object: %s", path)

    data = gcs.download(
        bucket_name=BUCKET_NAME,
        object_name=path
    )

    return pd.read_parquet(BytesIO(data))


def load_datasets(gcs):

    train_df = load_parquet(gcs, "train.parquet")

    validation_df = load_parquet(
        gcs, "validation.parquet"
    )

    test_df = load_parquet(gcs, "test.parquet")

    return train_df, validation_df, test_df


# ==========================================================
# 2. DATA VALIDATION
# ==========================================================

def validate_datasets(
    train_df,
    validation_df,
    test_df
):

    required_columns = (
        FEATURE_COLUMNS + [TARGET, "date"]
    )

    for name, df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df)
    ]:

        missing = set(required_columns) - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: {missing}"
            )

        if df[required_columns].isnull().any().any():
            raise ValueError(
                f"Null values found in {name} dataset"
            )

        numeric_data = df[FEATURE_COLUMNS + [TARGET]]

        if not np.isfinite(
            numeric_data.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Non-finite values found in {name} dataset"
            )

        if (df[TARGET] < 0).any():
            raise ValueError(
                f"Negative target values found in {name}"
            )

    # Check chronological separation
    train_end = pd.to_datetime(
        train_df["date"]
    ).max()

    val_start = pd.to_datetime(
        validation_df["date"]
    ).min()

    val_end = pd.to_datetime(
        validation_df["date"]
    ).max()

    test_start = pd.to_datetime(
        test_df["date"]
    ).min()

    if not train_end < val_start:
        raise ValueError(
            "Training and validation periods overlap"
        )

    if not val_end < test_start:
        raise ValueError(
            "Validation and test periods overlap"
        )

    logging.info("All datasets passed validation")


# ==========================================================
# 3. PREPARE X AND Y
# ==========================================================

def prepare_xy(df):

    X = df[FEATURE_COLUMNS].astype(float)

    y = df[TARGET].astype(float)

    return X, y


# ==========================================================
# 4. TRAIN XGBOOST MODEL
# ==========================================================

def train_model(
    X_train,
    y_train,
    X_validation,
    y_validation
):

    model = xgb.XGBRegressor(

        objective="reg:squarederror",

        n_estimators=MAX_ESTIMATORS,

        learning_rate=0.03,

        max_depth=2,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.1,

        reg_lambda=5.0,

        eval_metric="rmse",

        early_stopping_rounds=EARLY_STOPPING_ROUNDS,

        random_state=RANDOM_STATE,

        n_jobs=2,

        tree_method="hist"
    )

    logging.info("Starting XGBoost training")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_validation, y_validation)
        ],
        verbose=False
    )

    logging.info(
        "Best iteration: %s",
        model.best_iteration
    )

    logging.info(
        "Best validation RMSE: %s",
        model.best_score
    )

    return model


# ==========================================================
# 5. MODEL EVALUATION
# ==========================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    # Solar energy generation cannot be negative.
    # Apply the same constraint used in production.
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(
        y, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    r2 = r2_score(
        y, predictions
    )

    return {
        "MAE_kWh": float(mae),
        "RMSE_kWh": float(rmse),
        "R2": float(r2)
    }


# ==========================================================
# 6. FEATURE IMPORTANCE
# ==========================================================

def get_feature_importance(model):

    importance = model.feature_importances_

    result = {
        feature: float(value)
        for feature, value in zip(
            FEATURE_COLUMNS,
            importance
        )
    }

    return dict(
        sorted(
            result.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )


# ==========================================================
# 7. SAVE MODEL AND METRICS TO GCS
# ==========================================================

def save_model_artifacts(
    gcs,
    model,
    metrics,
    feature_importance,
    train_df,
    validation_df,
    test_df
):

    # Save native XGBoost JSON model
    with tempfile.TemporaryDirectory() as temp_dir:

        model_path = os.path.join(
            temp_dir,
            "solar_xgboost_model.json"
        )

        model.get_booster().save_model(model_path)

        with open(model_path, "rb") as f:
            model_bytes = f.read()

        gcs.upload(
            bucket_name=BUCKET_NAME,
            object_name=(
                f"{MODEL_PREFIX}/solar_xgboost_model.json"
            ),
            data=model_bytes,
            mime_type="application/json"
        )

    # Evaluation metrics
    metrics_payload = {
        "model": "XGBRegressor",
        "target": TARGET,
        "validation_metrics": metrics["validation"],
        "test_metrics": metrics["test"],
        "best_iteration": int(model.best_iteration),
        "feature_count": len(FEATURE_COLUMNS),
        "train_records": len(train_df),
        "validation_records": len(validation_df),
        "test_records": len(test_df),
        "evaluation_timestamp": datetime.utcnow().isoformat()
    }

    gcs.upload(
        bucket_name=BUCKET_NAME,
        object_name=f"{MODEL_PREFIX}/metrics.json",
        data=json.dumps(
            metrics_payload,
            indent=4
        ),
        mime_type="application/json"
    )

    # Feature importance
    gcs.upload(
        bucket_name=BUCKET_NAME,
        object_name=(
            f"{MODEL_PREFIX}/feature_importance.json"
        ),
        data=json.dumps(
            feature_importance,
            indent=4
        ),
        mime_type="application/json"
    )

    # Model metadata
    metadata = {
        "model_type": "XGBoost Regressor",
        "model_version": "1.0",
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "prediction_unit": "kWh/day",
        "best_iteration": int(model.best_iteration),
        "random_state": RANDOM_STATE,
        "input_source": FEATURE_PREFIX,
        "model_path": (
            f"{MODEL_PREFIX}/solar_xgboost_model.json"
        )
    }

    gcs.upload(
        bucket_name=BUCKET_NAME,
        object_name=f"{MODEL_PREFIX}/metadata.json",
        data=json.dumps(metadata, indent=4),
        mime_type="application/json"
    )

    logging.info("All model artifacts uploaded successfully")


# ==========================================================
# 8. MAIN TRAINING PIPELINE
# ==========================================================

def run_xgboost_training():

    gcs = GCSHook(
        gcp_conn_id=GCP_CONN_ID
    )

    # Load datasets
    train_df, validation_df, test_df = (
        load_datasets(gcs)
    )

    logging.info(
        "Train: %s | Validation: %s | Test: %s",
        len(train_df),
        len(validation_df),
        len(test_df)
    )

    # Validate datasets
    validate_datasets(
        train_df,
        validation_df,
        test_df
    )

    # Prepare train data
    X_train, y_train = prepare_xy(train_df)

    X_validation, y_validation = prepare_xy(
        validation_df
    )

    X_test, y_test = prepare_xy(test_df)

    # Train model using validation for early stopping
    model = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    # Evaluate validation
    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation
    )

    # Evaluate final test set only after training
    test_metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    metrics = {
        "validation": validation_metrics,
        "test": test_metrics
    }

    logging.info(
        "Validation metrics: %s",
        validation_metrics
    )

    logging.info(
        "Test metrics: %s",
        test_metrics
    )

    # Feature importance
    feature_importance = get_feature_importance(
        model
    )

    logging.info(
        "Feature importance: %s",
        feature_importance
    )

    # Save artifacts
    save_model_artifacts(
        gcs,
        model,
        metrics,
        feature_importance,
        train_df,
        validation_df,
        test_df
    )

    logging.info(
        "XGBoost training pipeline completed successfully"
    )


# ==========================================================
# 9. AIRFLOW DAG
# ==========================================================

default_args = {
    "owner": "solar-ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}


with DAG(

    dag_id="daily_solar_xgboost_training",

    default_args=default_args,

    description=(
        "Train and evaluate XGBoost model "
        "for daily solar energy prediction"
    ),

    start_date=datetime(2025, 1, 1),

    schedule_interval=None,

    catchup=False,

    tags=[
        "solar",
        "xgboost",
        "model-training",
        "daily-forecast"
    ]

) as dag:

    train_xgboost = PythonOperator(

        task_id="train_and_evaluate_xgboost",

        python_callable=run_xgboost_training

    )

    train_xgboost
