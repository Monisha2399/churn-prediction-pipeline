"""
tests/test_features.py
───────────────────────
Unit tests for feature builders:
  - TFIDFFeatureBuilder
  - sentiment aggregation
  - feature_store label builder
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.tfidf_features import TFIDFFeatureBuilder
from src.features.sentiment_features import score_text, aggregate_sentiment
from src.features.feature_store import build_labels


# ── TFIDFFeatureBuilder ───────────────────────────────────────────
class TestTFIDFFeatureBuilder:
    def _corpus(self):
        return pd.Series([
            "internet down for three days terrible service",
            "billing issue overcharged again",
            "how do i update payment method",
            "reset router connection problem",
            "cancel subscription unhappy with service",
        ] * 10)  # repeat to meet min_df=3

    def test_fit_transform_shape(self):
        builder = TFIDFFeatureBuilder(max_features=100, min_df=1)
        result  = builder.fit_transform(self._corpus())
        assert result.shape[0] == len(self._corpus())
        assert result.shape[1] <= 100

    def test_column_names_prefixed(self):
        builder = TFIDFFeatureBuilder(max_features=50, min_df=1)
        result  = builder.fit_transform(self._corpus())
        assert all(c.startswith("tfidf_") for c in result.columns)

    def test_transform_without_fit_raises(self):
        builder = TFIDFFeatureBuilder()
        with pytest.raises(RuntimeError):
            builder.transform(pd.Series(["some text"]))

    def test_transform_unseen_text(self):
        """Transform should not raise on text not seen during fit."""
        builder = TFIDFFeatureBuilder(max_features=50, min_df=1)
        builder.fit(self._corpus())
        result = builder.transform(pd.Series(["completely unseen words zxqw"]))
        assert result.shape[0] == 1

    def test_empty_string_handled(self):
        builder = TFIDFFeatureBuilder(max_features=50, min_df=1)
        corpus  = self._corpus()
        builder.fit(corpus)
        result  = builder.transform(pd.Series([""]))
        assert result.shape[0] == 1


# ── Sentiment scoring ─────────────────────────────────────────────
class TestSentimentScoring:
    def test_negative_text(self):
        score = score_text("This is absolutely terrible and I hate it")
        assert score < 0

    def test_positive_text(self):
        score = score_text("Great service, very happy and satisfied")
        assert score > 0

    def test_neutral_text(self):
        score = score_text("I would like to update my address")
        assert -1.0 <= score <= 1.0

    def test_empty_string(self):
        assert score_text("") == 0.0

    def test_returns_float(self):
        assert isinstance(score_text("some text"), float)


class TestAggregateSentiment:
    def _make_tickets(self):
        return pd.DataFrame({
            "customer_id":    ["C1", "C1", "C2"],
            "created_at":     [
                "2026-10-05T10:00:00+00:00",
                "2026-10-20T10:00:00+00:00",
                "2026-10-22T10:00:00+00:00",
            ],
            "sentiment_score": [-0.8, -0.5, 0.3],
        })

    def test_returns_dataframe(self):
        result = aggregate_sentiment(self._make_tickets(), pd.Timestamp("2026-11-01", tz="UTC"))
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        result = aggregate_sentiment(self._make_tickets(), pd.Timestamp("2026-11-01", tz="UTC"))
        for col in ["customer_id", "sentiment_mean", "sentiment_min", "sentiment_trend"]:
            assert col in result.columns

    def test_mean_in_range(self):
        result = aggregate_sentiment(self._make_tickets(), pd.Timestamp("2026-11-01", tz="UTC"))
        assert result["sentiment_mean"].between(-1, 1).all()

    def test_no_future_data_used(self):
        """Tickets after snapshot must be excluded."""
        tickets = pd.DataFrame({
            "customer_id":    ["C1"],
            "created_at":     ["2026-11-05T10:00:00+00:00"],  # after snapshot
            "sentiment_score": [0.9],
        })
        result = aggregate_sentiment(tickets, pd.Timestamp("2026-11-01", tz="UTC"))
        # C1 should not appear
        if not result.empty:
            assert "C1" not in result["customer_id"].values


# ── Label builder ─────────────────────────────────────────────────
class TestBuildLabels:
    def _make_customers(self):
        return pd.DataFrame({
            "customer_id": ["C1", "C2", "C3", "C4"],
            "churn_date": [
                "2026-11-15",   # within 30d window → label=1
                "2026-12-05",   # outside 30d window → label=0
                "2026-10-20",   # before snapshot → label=0
                None,           # never churned → label=0
            ],
        })

    def test_label_within_window(self):
        df = build_labels(self._make_customers(), pd.Timestamp("2026-11-01"))
        assert df.loc[df["customer_id"] == "C1", "label"].iloc[0] == 1

    def test_label_outside_window(self):
        df = build_labels(self._make_customers(), pd.Timestamp("2026-11-01"))
        assert df.loc[df["customer_id"] == "C2", "label"].iloc[0] == 0

    def test_label_past_churn(self):
        df = build_labels(self._make_customers(), pd.Timestamp("2026-11-01"))
        assert df.loc[df["customer_id"] == "C3", "label"].iloc[0] == 0

    def test_label_no_churn(self):
        df = build_labels(self._make_customers(), pd.Timestamp("2026-11-01"))
        assert df.loc[df["customer_id"] == "C4", "label"].iloc[0] == 0

    def test_no_leakage_future_data(self):
        """Label must only use churn_date, nothing after snapshot."""
        customers = pd.DataFrame({
            "customer_id": ["C5"],
            "churn_date":  ["2026-11-30"],  # within window
        })
        df = build_labels(customers, pd.Timestamp("2026-11-01"))
        # Should be 1 since 2026-11-30 is within (2026-11-01, 2026-12-01]
        assert df.loc[df["customer_id"] == "C5", "label"].iloc[0] == 1
