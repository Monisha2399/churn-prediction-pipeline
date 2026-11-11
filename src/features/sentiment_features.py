"""
src/features/sentiment_features.py
────────────────────────────────────
Sentiment analysis using NLTK VADER.
Aggregates compound scores per customer window:
  - mean sentiment
  - minimum sentiment (worst ticket)
  - sentiment trend (recent vs older)
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.preprocessing.cleaner import download_nltk_resources

download_nltk_resources()


# ── Analyser singleton ────────────────────────────────────────────
_SIA: SentimentIntensityAnalyzer | None = None


def _get_sia() -> SentimentIntensityAnalyzer:
    global _SIA
    if _SIA is None:
        _SIA = SentimentIntensityAnalyzer()
    return _SIA


# ── Per-ticket scoring ────────────────────────────────────────────
def score_text(text: str) -> float:
    """Return VADER compound score in [-1, 1] for a single text."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return _get_sia().polarity_scores(text)["compound"]


def score_ticket_df(df: pd.DataFrame, text_col: str = "body_bert") -> pd.DataFrame:
    """
    Add a ``sentiment_score`` column to a tickets DataFrame.

    Parameters
    ----------
    df       : tickets DataFrame (must have text_col and created_at)
    text_col : column containing cleaned ticket text

    Returns
    -------
    df with sentiment_score column added (in-place copy)
    """
    df = df.copy()
    df["sentiment_score"] = df[text_col].apply(score_text)
    logger.info(
        f"Scored {len(df):,} tickets — mean={df['sentiment_score'].mean():.3f}, "
        f"min={df['sentiment_score'].min():.3f}"
    )
    return df


# ── Per-customer aggregation ──────────────────────────────────────
def aggregate_sentiment(
    tickets: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    window_days: int = 30,
) -> pd.DataFrame:
    """
    Aggregate VADER scores per customer strictly before snapshot_date.

    Requires ``tickets`` to already have ``sentiment_score`` and ``created_at``
    (UTC-aware or naive, normalised to UTC).

    Returns
    -------
    DataFrame with columns:
        customer_id, sentiment_mean, sentiment_min, sentiment_trend
    """
    cutoff = pd.Timestamp(snapshot_date, tz="UTC")
    tickets["created_at"] = pd.to_datetime(tickets["created_at"], utc=True)
    past = tickets[tickets["created_at"] < cutoff].copy()

    if past.empty:
        logger.warning("No tickets found before snapshot — returning empty sentiment features.")
        return pd.DataFrame(columns=["customer_id", "sentiment_mean", "sentiment_min", "sentiment_trend"])

    window_start = cutoff - pd.Timedelta(days=window_days)
    recent = past[past["created_at"] >= window_start]
    older  = past[past["created_at"] <  window_start]

    rows = []
    for cid, grp in recent.groupby("customer_id"):
        mean_score = grp["sentiment_score"].mean()
        min_score  = grp["sentiment_score"].min()

        # Trend: recent mean minus older mean (negative trend = worsening)
        older_grp = older[older["customer_id"] == cid]
        older_mean = older_grp["sentiment_score"].mean() if not older_grp.empty else mean_score
        trend = mean_score - older_mean

        rows.append({
            "customer_id":      cid,
            "sentiment_mean":   round(mean_score, 4),
            "sentiment_min":    round(min_score, 4),
            "sentiment_trend":  round(trend, 4),
        })

    result = pd.DataFrame(rows)
    logger.info(
        f"Sentiment aggregated for {len(result):,} customers "
        f"(snapshot={snapshot_date}, window={window_days}d)"
    )
    return result
