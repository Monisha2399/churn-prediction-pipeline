"""
src/api/schemas.py
───────────────────
Pydantic v2 request / response schemas for the FastAPI prediction service.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request schemas ───────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    """
    Full feature payload for a single customer.
    All fields are optional — missing values are imputed to 0 server-side.
    """
    customer_id: str = Field(..., description="Unique customer identifier", examples=["CUST-001"])

    # CRM
    tenure_months:    Optional[float] = Field(None, ge=0)
    monthly_charges:  Optional[float] = Field(None, ge=0)
    contract_encoded: Optional[int]   = Field(None, ge=0, le=2,
                                               description="0=Month-to-month, 1=One year, 2=Two year")
    senior_citizen:   Optional[int]   = Field(None, ge=0, le=1)

    # Behavioural
    logins_30d:       Optional[float] = Field(None, ge=0)
    session_min_30d:  Optional[float] = Field(None, ge=0)
    login_trend:      Optional[float] = None
    failed_pay_30d:   Optional[int]   = Field(None, ge=0)
    days_since_login: Optional[int]   = Field(None, ge=0)

    # Ticket aggregates
    ticket_count_30d: Optional[int] = Field(None, ge=0)
    ticket_count_60d: Optional[int] = Field(None, ge=0)
    ticket_count_90d: Optional[int] = Field(None, ge=0)
    open_tickets:     Optional[int] = Field(None, ge=0)
    escalations_30d:  Optional[int] = Field(None, ge=0)
    days_since_ticket: Optional[int] = Field(None, ge=0)

    # Sentiment
    sentiment_mean:  Optional[float] = Field(None, ge=-1, le=1)
    sentiment_min:   Optional[float] = Field(None, ge=-1, le=1)
    sentiment_trend: Optional[float] = None


class PredictRequest(BaseModel):
    """Single-customer prediction request."""
    customer: CustomerFeatures
    threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Classification threshold. Defaults to 0.4.",
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request (up to 500 customers)."""
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=500)
    threshold: float = Field(default=0.4, ge=0.0, le=1.0)


# ── Response schemas ──────────────────────────────────────────────
class PredictionResult(BaseModel):
    customer_id:       str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_label:       int   = Field(..., ge=0, le=1,
                                     description="1 = likely to churn within 30 days")
    model_version:     str
    predicted_at:      datetime


class PredictResponse(BaseModel):
    """Single prediction response."""
    prediction: PredictionResult


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions:  List[PredictionResult]
    model_version: str
    count:         int


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
    uptime_seconds: float
