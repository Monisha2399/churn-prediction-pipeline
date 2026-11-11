"""
tests/test_api.py
──────────────────
Integration tests for the FastAPI prediction service.
Uses TestClient with a mocked model loader so no MLflow server is needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Mock model loader before importing app ────────────────────────
@pytest.fixture(autouse=True)
def mock_model_loader():
    """Patch model_loader so tests run without a real MLflow server."""
    with (
        patch("src.api.model_loader.load_champion_model"),
        patch("src.api.model_loader.is_model_loaded", return_value=True),
        patch("src.api.model_loader.get_model_version", return_value="v1"),
        patch("src.api.model_loader.get_uptime_seconds", return_value=42.0),
        patch("src.api.model_loader.predict_single", return_value=(0.73, 1)),
        patch(
            "src.api.model_loader.predict_batch",
            return_value=[(0.73, 1), (0.12, 0)],
        ),
    ):
        yield


@pytest.fixture
def client():
    from src.api.main import app
    return TestClient(app)


# ── /health ───────────────────────────────────────────────────────
class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_model_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_has_model_version(self, client):
        data = client.get("/health").json()
        assert "model_version" in data


# ── /predict ─────────────────────────────────────────────────────
class TestPredict:
    def _payload(self):
        return {
            "customer": {
                "customer_id":      "CUST-001",
                "tenure_months":    12.0,
                "monthly_charges":  80.5,
                "contract_encoded": 0,
                "logins_30d":       5.0,
                "sentiment_mean":   -0.4,
            },
            "threshold": 0.4,
        }

    def test_returns_200(self, client):
        resp = client.post("/predict", json=self._payload())
        assert resp.status_code == 200

    def test_has_prediction_keys(self, client):
        data = client.post("/predict", json=self._payload()).json()
        pred = data["prediction"]
        assert "customer_id"       in pred
        assert "churn_probability" in pred
        assert "churn_label"       in pred
        assert "model_version"     in pred

    def test_customer_id_matches(self, client):
        data = client.post("/predict", json=self._payload()).json()
        assert data["prediction"]["customer_id"] == "CUST-001"

    def test_probability_in_range(self, client):
        data = client.post("/predict", json=self._payload()).json()
        prob = data["prediction"]["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_label_is_binary(self, client):
        data = client.post("/predict", json=self._payload()).json()
        assert data["prediction"]["churn_label"] in (0, 1)

    def test_missing_optional_fields_ok(self, client):
        """All feature fields except customer_id are optional."""
        payload = {"customer": {"customer_id": "CUST-002"}, "threshold": 0.5}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_invalid_threshold_rejected(self, client):
        payload = self._payload()
        payload["threshold"] = 1.5   # out of range
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


# ── /predict/batch ────────────────────────────────────────────────
class TestBatchPredict:
    def _payload(self):
        return {
            "customers": [
                {"customer_id": "CUST-001", "tenure_months": 12.0},
                {"customer_id": "CUST-002", "tenure_months": 36.0},
            ],
            "threshold": 0.4,
        }

    def test_returns_200(self, client):
        resp = client.post("/predict/batch", json=self._payload())
        assert resp.status_code == 200

    def test_count_matches(self, client):
        data = client.post("/predict/batch", json=self._payload()).json()
        assert data["count"] == 2

    def test_predictions_list_length(self, client):
        data = client.post("/predict/batch", json=self._payload()).json()
        assert len(data["predictions"]) == 2

    def test_empty_list_rejected(self, client):
        resp = client.post("/predict/batch", json={"customers": []})
        assert resp.status_code == 422

    def test_over_limit_rejected(self, client):
        customers = [{"customer_id": f"C{i}"} for i in range(501)]
        resp = client.post("/predict/batch", json={"customers": customers})
        assert resp.status_code == 422
