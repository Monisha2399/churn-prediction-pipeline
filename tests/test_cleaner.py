"""
tests/test_cleaner.py
──────────────────────
Unit tests for src/preprocessing/cleaner.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.cleaner import (
    aggregate_tickets,
    clean_text_for_bert,
    clean_text_for_tfidf,
    clean_ticket_df,
    preprocess_crm,
    preprocess_events,
)


# ── clean_text_for_bert ───────────────────────────────────────────
class TestCleanTextForBert:
    def test_strips_html(self):
        assert "<b>" not in clean_text_for_bert("<b>Hello</b>")

    def test_removes_url(self):
        result = clean_text_for_bert("Visit http://example.com for help")
        assert "http" not in result

    def test_removes_email(self):
        result = clean_text_for_bert("Contact support@example.com please")
        assert "@" not in result

    def test_removes_ticket_id(self):
        result = clean_text_for_bert("Ref TKT-12345: your issue")
        assert "TKT" not in result
        assert "12345" not in result

    def test_normalises_whitespace(self):
        result = clean_text_for_bert("too   many    spaces")
        assert "  " not in result

    def test_empty_string(self):
        assert clean_text_for_bert("") == ""

    def test_none_returns_empty(self):
        assert clean_text_for_bert(None) == ""  # type: ignore

    def test_preserves_case(self):
        result = clean_text_for_bert("Hello World")
        assert "Hello" in result


# ── clean_text_for_tfidf ─────────────────────────────────────────
class TestCleanTextForTfidf:
    def test_lowercases(self):
        result = clean_text_for_tfidf("UPPER CASE TEXT")
        assert result == result.lower()

    def test_removes_stopwords(self):
        result = clean_text_for_tfidf("the internet is down")
        assert "the" not in result.split()
        assert "is" not in result.split()

    def test_removes_punctuation(self):
        result = clean_text_for_tfidf("Hello, World! How are you?")
        assert "," not in result
        assert "!" not in result

    def test_returns_string(self):
        assert isinstance(clean_text_for_tfidf("some text"), str)


# ── clean_ticket_df ───────────────────────────────────────────────
class TestCleanTicketDf:
    def _make_df(self):
        return pd.DataFrame({
            "ticket_id":   [1, 2],
            "customer_id": ["C1", "C2"],
            "created_at":  ["2026-10-01T10:00:00", "2026-10-15T12:00:00"],
            "subject":     ["Billing issue", "Technical problem"],
            "body":        ["I was charged twice", "Internet is down"],
            "status":      ["open", "resolved"],
        })

    def test_adds_body_bert(self):
        df = clean_ticket_df(self._make_df())
        assert "body_bert" in df.columns

    def test_adds_body_tfidf(self):
        df = clean_ticket_df(self._make_df())
        assert "body_tfidf" in df.columns

    def test_created_at_is_datetime(self):
        df = clean_ticket_df(self._make_df())
        assert pd.api.types.is_datetime64_any_dtype(df["created_at"])

    def test_no_null_in_body_bert(self):
        df = clean_ticket_df(self._make_df())
        assert df["body_bert"].notna().all()


# ── aggregate_tickets ─────────────────────────────────────────────
class TestAggregateTickets:
    def _make_tickets(self):
        return pd.DataFrame({
            "customer_id": ["C1", "C1", "C2"],
            "created_at":  [
                "2026-10-05T10:00:00+00:00",
                "2026-10-20T10:00:00+00:00",
                "2026-10-25T10:00:00+00:00",
            ],
            "status":     ["open", "resolved", "escalated"],
            "body_bert":  ["bad service", "still bad", "very bad"],
            "body_tfidf": ["bad service", "still bad", "very bad"],
        })

    def test_returns_dataframe(self):
        tickets = self._make_tickets()
        result = aggregate_tickets(tickets, pd.Timestamp("2026-11-01", tz="UTC"))
        assert isinstance(result, pd.DataFrame)

    def test_has_ticket_count_cols(self):
        tickets = self._make_tickets()
        result = aggregate_tickets(tickets, pd.Timestamp("2026-11-01", tz="UTC"))
        assert "ticket_count_30d" in result.columns

    def test_no_future_data(self):
        """Tickets after snapshot should not be counted."""
        tickets = pd.DataFrame({
            "customer_id": ["C1"],
            "created_at":  ["2026-11-05T10:00:00+00:00"],   # AFTER snapshot
            "status":      ["open"],
            "body_bert":   ["future ticket"],
            "body_tfidf":  ["future ticket"],
        })
        result = aggregate_tickets(tickets, pd.Timestamp("2026-11-01", tz="UTC"))
        # C1 should not appear (all tickets are in the future)
        if not result.empty:
            c1 = result[result["customer_id"] == "C1"]
            assert c1.empty or int(c1["ticket_count_30d"].iloc[0]) == 0


# ── preprocess_crm ────────────────────────────────────────────────
class TestPreprocessCrm:
    def _make_crm(self):
        return pd.DataFrame({
            "customer_id":    ["C1", "C2"],
            "contract_type":  ["Month-to-month", "Two year"],
            "monthly_charges": [70.5, 99.0],
            "total_charges":   ["1000.0", " "],
            "tenure_months":   [12, 24],
            "senior_citizen":  [0, 1],
        })

    def test_contract_encoded_values(self):
        df = preprocess_crm(self._make_crm())
        assert df.loc[df["customer_id"] == "C1", "contract_encoded"].iloc[0] == 0
        assert df.loc[df["customer_id"] == "C2", "contract_encoded"].iloc[0] == 2

    def test_total_charges_numeric(self):
        df = preprocess_crm(self._make_crm())
        assert pd.api.types.is_float_dtype(df["total_charges"])

    def test_no_nulls_in_numeric(self):
        df = preprocess_crm(self._make_crm())
        for col in ["monthly_charges", "total_charges", "tenure_months"]:
            assert df[col].notna().all()
