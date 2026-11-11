"""
src/features/bert_embeddings.py
────────────────────────────────
BERT-style sentence embeddings using sentence-transformers.
Model: sentence-transformers/all-MiniLM-L6-v2  (22M params, 384-dim)
  → fast on CPU, free-tier compatible.

Embeddings are computed once and cached to S3 as Parquet to avoid
re-embedding on every training run.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Lazy import — only pull in torch/transformers when actually called
_MODEL = None
_MODEL_NAME = os.environ.get(
    "BERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)


def _get_model():
    """Lazy-load the sentence-transformer model (singleton)."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {_MODEL_NAME}")
        _MODEL = SentenceTransformer(_MODEL_NAME)
        logger.info("Model loaded.")
    return _MODEL


# ── Core embedding function ───────────────────────────────────────
def embed_texts(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of strings to a numpy array of shape (n, 384).

    Parameters
    ----------
    texts        : list of cleaned text strings
    batch_size   : encoding batch size (reduce if OOM)
    show_progress: show tqdm bar

    Returns
    -------
    np.ndarray of float32, shape (len(texts), embedding_dim)
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity ready
    )
    logger.info(f"Embedded {len(texts):,} texts → shape {embeddings.shape}")
    return embeddings.astype(np.float32)


# ── Per-customer mean pooling ─────────────────────────────────────
def build_customer_embeddings(
    tickets: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    window_days: int = 30,
    text_col: str = "body_bert",
) -> pd.DataFrame:
    """
    Mean-pool ticket embeddings per customer within [snapshot - window, snapshot).

    Parameters
    ----------
    tickets       : cleaned tickets DataFrame with customer_id, created_at, body_bert
    snapshot_date : cutoff date (no data after this is used)
    window_days   : look-back window
    text_col      : column with lightly-cleaned text (for BERT)

    Returns
    -------
    DataFrame: customer_id + 384 embedding columns (emb_0 … emb_383)
    """
    cutoff = pd.Timestamp(snapshot_date, tz="UTC")
    tickets["created_at"] = pd.to_datetime(tickets["created_at"], utc=True)

    window_start = cutoff - pd.Timedelta(days=window_days)
    recent = tickets[
        (tickets["created_at"] >= window_start) &
        (tickets["created_at"] <  cutoff)
    ].copy()

    if recent.empty:
        logger.warning("No tickets in window — returning empty embeddings.")
        return pd.DataFrame()

    # Concatenate all ticket text per customer
    customer_texts = (
        recent.groupby("customer_id")[text_col]
        .apply(lambda s: " ".join(s.fillna("").tolist()))
        .reset_index()
    )
    customer_texts.columns = ["customer_id", "combined_text"]

    texts = customer_texts["combined_text"].tolist()
    emb_matrix = embed_texts(texts)

    dim = emb_matrix.shape[1]
    emb_cols = {f"emb_{i}": emb_matrix[:, i] for i in range(dim)}
    emb_df = pd.DataFrame(emb_cols)
    emb_df.insert(0, "customer_id", customer_texts["customer_id"].values)

    logger.info(
        f"Customer embeddings: {len(emb_df):,} customers × {dim} dims "
        f"(snapshot={snapshot_date})"
    )
    return emb_df


# ── S3 Cache helpers ──────────────────────────────────────────────
def _cache_key(snapshot_date: pd.Timestamp, window_days: int) -> str:
    date_str = str(snapshot_date.date()) if hasattr(snapshot_date, "date") else str(snapshot_date)
    return f"features/embeddings/embeddings_{date_str}_w{window_days}.parquet"


def load_or_build_embeddings(
    tickets: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    window_days: int = 30,
    bucket: Optional[str] = None,
    text_col: str = "body_bert",
) -> pd.DataFrame:
    """
    Return cached embeddings from S3 if available, otherwise compute and cache.

    Parameters
    ----------
    tickets       : cleaned tickets DataFrame
    snapshot_date : snapshot cutoff date
    window_days   : lookback window
    bucket        : S3 bucket (defaults to env var S3_BUCKET)
    text_col      : text column for embedding

    Returns
    -------
    DataFrame: customer_id + emb_0 … emb_383
    """
    from src.ingestion.s3_loader import key_exists, load_parquet_from_s3, upload_dataframe_to_s3

    _bucket = bucket or os.environ.get("S3_BUCKET", "churn-bucket")
    cache_key = _cache_key(snapshot_date, window_days)

    if key_exists(cache_key, bucket=_bucket):
        logger.info(f"Cache hit — loading embeddings from s3://{_bucket}/{cache_key}")
        return load_parquet_from_s3(cache_key, bucket=_bucket)

    logger.info("Cache miss — computing embeddings …")
    emb_df = build_customer_embeddings(tickets, snapshot_date, window_days, text_col)

    if not emb_df.empty:
        upload_dataframe_to_s3(emb_df, cache_key, bucket=_bucket, fmt="parquet")

    return emb_df
