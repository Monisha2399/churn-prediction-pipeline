"""
src/api/main.py
────────────────
FastAPI prediction service for customer churn.

Endpoints:
  GET  /health                — liveness + model status
  POST /predict               — single customer prediction
  POST /predict/batch         — batch prediction (up to 500 customers)

Metrics exposed at /metrics for Prometheus scraping.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, status
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.model_loader import (
    get_model_version,
    get_uptime_seconds,
    is_model_loaded,
    load_champion_model,
    predict_batch,
    predict_single,
)
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionResult,
)


# ── Lifespan (startup / shutdown) ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model once at startup."""
    logger.info("FastAPI starting up — loading champion model …")
    try:
        load_champion_model()
        logger.info("Model ready.")
    except Exception as exc:
        logger.error(f"Model load failed: {exc}. /predict will return 503 until fixed.")
    yield
    logger.info("FastAPI shutting down.")


def _ensure_model() -> bool:
    """Try to (re)load the champion if it was not available at startup."""
    if not is_model_loaded():
        try:
            load_champion_model()
        except Exception as exc:
            logger.warning(f"Champion model still unavailable: {exc}")
    return is_model_loaded()


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts customer churn 30 days in advance using CRM, "
        "behavioural and NLP (BERT) features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)


# ── Routes ────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Liveness check and model status."""
    return HealthResponse(
        status="ok" if is_model_loaded() else "degraded",
        model_loaded=is_model_loaded(),
        model_version=get_model_version(),
        uptime_seconds=round(get_uptime_seconds(), 1),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict churn probability for a single customer.

    Returns churn_probability ∈ [0,1] and churn_label (0 = stay, 1 = churn).
    """
    if not _ensure_model():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Retry in a few seconds.",
        )

    feature_dict = request.customer.model_dump()
    customer_id  = feature_dict.pop("customer_id")

    try:
        prob, label = predict_single(feature_dict, threshold=request.threshold)
    except Exception as exc:
        logger.error(f"Prediction error for {customer_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )

    return PredictResponse(
        prediction=PredictionResult(
            customer_id=customer_id,
            churn_probability=round(prob, 4),
            churn_label=label,
            model_version=get_model_version(),
            predicted_at=datetime.now(timezone.utc),
        )
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch_endpoint(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Predict churn for a batch of customers (max 500 per request).
    """
    if not _ensure_model():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Retry in a few seconds.",
        )

    feature_dicts = [c.model_dump() for c in request.customers]
    customer_ids  = [d.pop("customer_id") for d in feature_dicts]

    try:
        results = predict_batch(feature_dicts, threshold=request.threshold)
    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        )

    now = datetime.now(timezone.utc)
    version = get_model_version()

    predictions = [
        PredictionResult(
            customer_id=cid,
            churn_probability=round(prob, 4),
            churn_label=label,
            model_version=version,
            predicted_at=now,
        )
        for cid, (prob, label) in zip(customer_ids, results)
    ]

    return BatchPredictResponse(
        predictions=predictions,
        model_version=version,
        count=len(predictions),
    )
