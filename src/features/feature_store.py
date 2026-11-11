"""
src/features/feature_store.py
──────────────────────────────
Joins structured (CRM + behavioural) and unstructured (NLP) features
into a single feature table for a given snapshot date.
Also builds the label column (churn within 30 days of snapshot).

Label rule (NO LEAKAGE):
    label = 1  if customer's churn_date is in (snapshot_date, snapshot_date + 30 days]
    Features use ONLY data strictly before snapshot_date.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from loguru import logger

from src.features.bert_embeddings import load_or_build_embeddings
from src.features.sentiment_features import aggregate_sentiment, score_ticket_df
from src.features.tfidf_features import TFIDFFeatureBuilder
from src.preprocessing.cleaner import (
    aggregate_tickets,
    clean_ticket_df,
    preprocess_crm,
    preprocess_events,
)


# ── Label builder ─────────────────────────────────────────────────
def build_labels(
    customers: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """
    Return a DataFrame with customer_id and binary label.

    label = 1 if churn_date ∈ (snapshot_date, snapshot_date + horizon_days].
    Uses only churn_date from CRM — never any data after snapshot_date.
    """
    snap = pd.Timestamp(snapshot_date)
    horizon_end = snap + pd.Timedelta(days=horizon_days)

    df = customers[["customer_id", "churn_date"]].copy()
    df["churn_date"] = pd.to_datetime(df["churn_date"])

    df["label"] = (
        (df["churn_date"] > snap) & (df["churn_date"] <= horizon_end)
    ).astype(int)

    churn_rate = df["label"].mean()
    logger.info(
        f"Labels built for snapshot={snap.date()} | "
        f"n={len(df):,} | churn_rate={churn_rate:.1%}"
    )
    return df[["customer_id", "label"]]


def active_at(customers: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Customers who had signed up and not yet churned at snapshot T (the only valid population)."""
    snap = pd.Timestamp(snapshot_date)
    mask = pd.Series(True, index=customers.index)
    if "signup_date" in customers.columns:
        mask &= pd.to_datetime(customers["signup_date"]) <= snap
    mask &= pd.to_datetime(customers["churn_date"]).isna() | (pd.to_datetime(customers["churn_date"]) > snap)
    return customers[mask]


# ── Main feature builder ──────────────────────────────────────────
def build_feature_table(
    customers: pd.DataFrame,
    tickets: pd.DataFrame,
    events: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    tfidf_builder: Optional[TFIDFFeatureBuilder] = None,
    include_bert: bool = True,
    include_tfidf: bool = True,
    bucket: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the full feature table for one snapshot date.

    Parameters
    ----------
    customers     : raw customers DataFrame
    tickets       : raw tickets DataFrame
    events        : raw events DataFrame
    snapshot_date : reference date T; all features are computed before T
    tfidf_builder : fitted TFIDFFeatureBuilder (if None, TF-IDF is skipped)
    include_bert  : whether to add BERT embedding columns
    include_tfidf : whether to add TF-IDF columns
    bucket        : S3 bucket for embedding cache

    Returns
    -------
    DataFrame indexed by customer_id with all features + label
    """
    snap = pd.Timestamp(snapshot_date)
    logger.info(f"Building feature table for snapshot {snap.date()} …")

    # ── Labels (only customers still active at T) ─────────────────
    customers = active_at(customers, snap)
    labels = build_labels(customers, snap)

    # ── CRM features ──────────────────────────────────────────────
    crm = preprocess_crm(customers)[
        ["customer_id", "tenure_months", "monthly_charges", "contract_encoded", "senior_citizen"]
    ]

    # ── Behavioural features ──────────────────────────────────────
    behav = preprocess_events(events, snap)

    # ── Ticket cleaning + aggregation ─────────────────────────────
    tickets_clean = clean_ticket_df(tickets)
    ticket_agg    = aggregate_tickets(tickets_clean, snap)

    # ── Sentiment ─────────────────────────────────────────────────
    tickets_scored = score_ticket_df(tickets_clean)
    sentiment      = aggregate_sentiment(tickets_scored, snap)

    # ── Join everything ───────────────────────────────────────────
    feat = (
        labels
        .merge(crm,        on="customer_id", how="left")
        .merge(behav,      on="customer_id", how="left")
        .merge(ticket_agg.drop(columns=["text_bert", "text_tfidf"], errors="ignore"),
               on="customer_id", how="left")
        .merge(sentiment,  on="customer_id", how="left")
    )

    # ── TF-IDF features ───────────────────────────────────────────
    if include_tfidf and tfidf_builder is not None and "text_tfidf" in ticket_agg.columns:
        text_map = ticket_agg.set_index("customer_id")["text_tfidf"]
        aligned_texts = feat["customer_id"].map(text_map).fillna("")
        tfidf_feats = tfidf_builder.transform(aligned_texts)
        tfidf_feats.index = feat.index
        feat = pd.concat([feat, tfidf_feats], axis=1)
        logger.info(f"TF-IDF: added {tfidf_feats.shape[1]} columns")

    # ── BERT embedding features ───────────────────────────────────
    if include_bert:
        emb_df = load_or_build_embeddings(
            tickets_clean, snap, window_days=30, bucket=bucket
        )
        if not emb_df.empty:
            feat = feat.merge(emb_df, on="customer_id", how="left")
            logger.info(f"BERT: added {len([c for c in feat.columns if c.startswith('emb_')])} embedding columns")

    # ── Fill remaining NaNs ────────────────────────────────────────
    num_cols = feat.select_dtypes(include="number").columns
    feat[num_cols] = feat[num_cols].fillna(0)

    feat["snapshot_date"] = snap.date()

    logger.info(
        f"Feature table ready: {feat.shape[0]:,} rows × {feat.shape[1]} columns | "
        f"label balance: {feat['label'].mean():.1%}"
    )
    return feat


# ── Train / test split (TIME-BASED — no random split) ─────────────
def time_based_split(
    feature_tables: list[pd.DataFrame],
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenate per-snapshot feature tables and split by snapshot_date.
    Earlier snapshots → train; latest snapshots → test.
    This is the ONLY valid split for time-series churn data.

    Parameters
    ----------
    feature_tables : list of DataFrames, one per snapshot
    test_fraction  : fraction of snapshots to use as test set

    Returns
    -------
    (train_df, test_df)
    """
    combined = pd.concat(feature_tables, ignore_index=True)
    combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"])
    dates = sorted(combined["snapshot_date"].unique())

    n_test = max(1, int(len(dates) * test_fraction))
    test_dates = dates[-n_test:]
    train_dates = dates[:-n_test]

    train = combined[combined["snapshot_date"].isin(train_dates)].copy()
    test  = combined[combined["snapshot_date"].isin(test_dates)].copy()

    logger.info(
        f"Time-based split: train={len(train):,} (snapshots {[str(d.date()) for d in train_dates]}) | "
        f"test={len(test):,} (snapshots {[str(d.date()) for d in test_dates]})"
    )
    return train, test
