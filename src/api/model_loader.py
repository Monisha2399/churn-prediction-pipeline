"""
src/api/model_loader.py
────────────────────────
Loads the champion model from the MLflow Model Registry at startup.
Exposes a module-level singleton so the pipeline is loaded once and
reused across all requests.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger

# ── Config ────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI",  "http://mlflow:5000")
CHAMPION_MODEL_NAME  = os.environ.get("CHAMPION_MODEL_NAME",  "churn_classifier")
CHAMPION_ALIAS       = os.environ.get("MODEL_NAME",           "champion")

# ── Singleton state ───────────────────────────────────────────────
_pipeline       = None
_model_version  = "unknown"
_load_time: float = 0.0


def load_champion_model() -> None:
    """
    Download and cache the @champion model from the MLflow registry.
    Called once at FastAPI startup.
    """
    global _pipeline, _model_version, _load_time

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{CHAMPION_MODEL_NAME}@{CHAMPION_ALIAS}"

    logger.info(f"Loading model from: {model_uri}")
    start = time.perf_counter()
    _pipeline = mlflow.sklearn.load_model(model_uri)
    elapsed = time.perf_counter() - start
    _load_time = time.time()

    # Resolve version string
    client = mlflow.tracking.MlflowClient()
    try:
        version_info = client.get_model_version_by_alias(CHAMPION_MODEL_NAME, CHAMPION_ALIAS)
        _model_version = f"v{version_info.version}"
    except Exception:
        _model_version = "unknown"

    logger.info(f"Model loaded in {elapsed:.2f}s — version={_model_version}")


def get_pipeline():
    """Return the cached pipeline (raises if not loaded)."""
    if _pipeline is None:
        raise RuntimeError("Model not loaded. Call load_champion_model() first.")
    return _pipeline


def get_model_version() -> str:
    return _model_version


def is_model_loaded() -> bool:
    return _pipeline is not None


def get_uptime_seconds() -> float:
    if _load_time == 0:
        return 0.0
    return time.time() - _load_time


# ── Inference helpers ─────────────────────────────────────────────
def _align(df: pd.DataFrame) -> pd.DataFrame:
    """Select/reorder columns exactly as the model was trained; missing -> 0."""
    names = getattr(get_pipeline(), "feature_names_in_", None)
    if names is not None:
        df = df.reindex(columns=list(names), fill_value=0)
    return df.fillna(0).astype(float)


def features_to_dataframe(feature_dict: dict) -> pd.DataFrame:
    """
    Convert a CustomerFeatures dict to a one-row DataFrame
    aligned with the training feature columns.
    Missing values are filled with 0 (same imputation as training).
    """
    df = pd.DataFrame([feature_dict])
    # Drop non-feature columns
    df = df.drop(columns=["customer_id"], errors="ignore")
    return _align(df)


def predict_single(feature_dict: dict, threshold: float = 0.4) -> tuple[float, int]:
    """
    Run inference for one customer.

    Returns
    -------
    (churn_probability, churn_label)
    """
    pipeline = get_pipeline()
    X = features_to_dataframe(feature_dict)
    prob  = float(pipeline.predict_proba(X)[0, 1])
    label = int(prob >= threshold)
    return prob, label


def predict_batch(feature_dicts: list[dict], threshold: float = 0.4) -> list[tuple[float, int]]:
    """
    Run inference for a batch of customers.

    Returns
    -------
    list of (churn_probability, churn_label)
    """
    pipeline = get_pipeline()
    df = _align(pd.DataFrame(feature_dicts).drop(columns=["customer_id"], errors="ignore"))
    probs  = pipeline.predict_proba(df)[:, 1]
    labels = (probs >= threshold).astype(int)
    return list(zip(probs.tolist(), labels.tolist()))
