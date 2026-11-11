"""
src/preprocessing/cleaner.py
─────────────────────────────
Text cleaning, tokenisation, lemmatisation (NLTK) and structured-data
preprocessing (encoding, scaling, missing-value imputation).

Design notes:
- Raw text is cleaned lightly for BERT (strip HTML, normalise whitespace).
- Heavy NLP cleaning (stopwords, lemmatisation) is a separate path for
  TF-IDF, because BERT works best on near-original text.
- No data from after snapshot_date is ever used here.
"""

from __future__ import annotations

import re
import string
from functools import lru_cache
from typing import List, Optional

import nltk
import pandas as pd
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ── NLTK resource bootstrap ───────────────────────────────────────
_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("sentiment/vader_lexicon", "vader_lexicon"),
]


def download_nltk_resources() -> None:
    """Download required NLTK data packages (idempotent)."""
    for path, pkg in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {pkg}")
            nltk.download(pkg, quiet=True)


download_nltk_resources()


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _stop_words() -> set:
    return set(stopwords.words("english"))


# ── Text Cleaning ─────────────────────────────────────────────────
_HTML_TAG_RE    = re.compile(r"<[^>]+>")
_URL_RE         = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE       = re.compile(r"\S+@\S+\.\S+")
_TICKET_ID_RE   = re.compile(r"\b(TKT|TICKET|REF|#)\s*[-:#]?\s*\d+\b", re.I)
_WHITESPACE_RE  = re.compile(r"\s+")


def clean_text_for_bert(text: str) -> str:
    """
    Light cleaning suitable as BERT input.
    Strips HTML, URLs, email addresses and ticket reference IDs;
    normalises whitespace. Preserves case and punctuation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _TICKET_ID_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_text_for_tfidf(text: str) -> str:
    """
    Heavy cleaning for TF-IDF / bag-of-words models.
    Lowercases, removes punctuation, stopwords and lemmatises.
    """
    text = clean_text_for_bert(text).lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    lemma = _lemmatizer()
    stops = _stop_words()
    tokens = [lemma.lemmatize(t) for t in tokens if t not in stops and len(t) > 2]
    return " ".join(tokens)


def clean_ticket_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply both cleaning strategies to a tickets DataFrame.

    Expected columns: ticket_id, customer_id, created_at, body, subject, …
    Adds: body_bert, body_tfidf
    """
    df = df.copy()
    df["body"] = df["body"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)

    combined = df["subject"] + " " + df["body"]
    df["body_bert"]  = combined.apply(clean_text_for_bert)
    df["body_tfidf"] = combined.apply(clean_text_for_tfidf)

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    logger.info(f"Cleaned {len(df):,} ticket rows.")
    return df


# ── Ticket aggregation per customer × snapshot ────────────────────
def aggregate_tickets(
    tickets: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    windows: list[int] = [30, 60, 90],
) -> pd.DataFrame:
    """
    Aggregate ticket features per customer strictly before snapshot_date.

    Returns one row per customer with columns:
        ticket_count_{w}d, open_tickets, escalations_30d,
        days_since_ticket, text_bert, text_tfidf
    (text columns are concatenations of recent ticket bodies)
    """
    cutoff = pd.Timestamp(snapshot_date, tz="UTC")
    past = tickets[tickets["created_at"] < cutoff].copy()

    if past.empty:
        return pd.DataFrame()

    rows = []
    for cid, grp in past.groupby("customer_id"):
        row: dict = {"customer_id": cid}
        for w in windows:
            window_start = cutoff - pd.Timedelta(days=w)
            sub = grp[grp["created_at"] >= window_start]
            row[f"ticket_count_{w}d"] = len(sub)

        # 30-day window details
        w30 = grp[grp["created_at"] >= cutoff - pd.Timedelta(days=30)]
        row["open_tickets"]    = int((w30.get("status", pd.Series()) == "open").sum())
        row["escalations_30d"] = int((w30.get("status", pd.Series()) == "escalated").sum())

        last_ticket = grp["created_at"].max()
        row["days_since_ticket"] = (cutoff - last_ticket).days

        # Concatenate recent text for NLP features
        row["text_bert"]  = " ".join(w30["body_bert"].tolist())
        row["text_tfidf"] = " ".join(w30["body_tfidf"].tolist())
        rows.append(row)

    result = pd.DataFrame(rows)
    logger.debug(f"Aggregated tickets for snapshot {snapshot_date.date()} → {len(result)} customers")
    return result


# ── Structured data preprocessing ────────────────────────────────
_ORDINAL_CONTRACT = {"Month-to-month": 0, "One year": 1, "Two year": 2}


def preprocess_crm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode and impute CRM (customers) table.
    Returns a copy with numeric columns ready for modelling.
    """
    df = df.copy()
    df["contract_encoded"] = (
        df["contract_type"]
        .map(_ORDINAL_CONTRACT)
        .fillna(0)
        .astype(int)
    )
    df["senior_citizen"] = df["senior_citizen"].fillna(0).astype(int)
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce").fillna(0.0)
    df["total_charges"]   = pd.to_numeric(df["total_charges"],   errors="coerce").fillna(0.0)
    df["tenure_months"]   = pd.to_numeric(df["tenure_months"],   errors="coerce").fillna(0.0)
    return df


def preprocess_events(
    events: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    window: int = 30,
) -> pd.DataFrame:
    """
    Aggregate behavioural events per customer strictly before snapshot_date.

    Returns one row per customer with columns:
        logins_30d, session_min_30d, login_trend, failed_pay_30d, days_since_login
    """
    cutoff = pd.Timestamp(snapshot_date).normalize()
    events["event_date"] = pd.to_datetime(events["event_date"])
    past = events[events["event_date"] < cutoff].copy()

    if past.empty:
        return pd.DataFrame()

    window_start = cutoff - pd.Timedelta(days=window)
    recent = past[past["event_date"] >= window_start]

    # Aggregations
    agg = (
        recent.groupby("customer_id")
        .agg(
            logins_30d    =("logins", "sum"),
            session_min_30d=("session_minutes", "sum"),
            failed_pay_30d=("failed_payments", "sum"),
        )
        .reset_index()
    )

    # Days since last login
    last_login = (
        past.groupby("customer_id")["event_date"]
        .max()
        .reset_index()
        .rename(columns={"event_date": "last_login"})
    )
    last_login["days_since_login"] = (cutoff - last_login["last_login"]).dt.days

    # Week-on-week login trend
    w1 = past[past["event_date"] >= cutoff - pd.Timedelta(days=7)].groupby("customer_id")["logins"].sum()
    w2 = past[
        (past["event_date"] >= cutoff - pd.Timedelta(days=14)) &
        (past["event_date"] <  cutoff - pd.Timedelta(days=7))
    ].groupby("customer_id")["logins"].sum()
    trend = ((w1 - w2) / (w2 + 1)).rename("login_trend").reset_index()

    result = (
        agg
        .merge(last_login[["customer_id", "days_since_login"]], on="customer_id", how="left")
        .merge(trend, on="customer_id", how="left")
    )
    result["login_trend"] = result["login_trend"].fillna(0.0)
    result["days_since_login"] = result["days_since_login"].fillna(999).astype(int)

    logger.debug(f"Aggregated events for snapshot {snapshot_date} → {len(result)} customers")
    return result
