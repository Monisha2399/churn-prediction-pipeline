"""
src/ingestion/db_loader.py
──────────────────────────
Load DataFrames into PostgreSQL using SQLAlchemy.
All writes are idempotent: upsert (INSERT … ON CONFLICT DO UPDATE)
for customers / tickets / events, truncate-and-load for features /
predictions.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ── Connection ────────────────────────────────────────────────────
def _build_dsn() -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db   = os.environ.get("POSTGRES_DB", "churndb")
    user = os.environ.get("POSTGRES_USER", "churnuser")
    pwd  = os.environ.get("POSTGRES_PASSWORD", "changeme_strong_password")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine (pool_pre_ping keeps idle connections alive)."""
    dsn = _build_dsn()
    return create_engine(dsn, pool_pre_ping=True, pool_size=5, max_overflow=10)


@contextmanager
def get_connection() -> Generator:
    engine = get_engine()
    with engine.connect() as conn:
        yield conn


# ── Schema initialisation ─────────────────────────────────────────
def init_schema(schema_path: str = "src/ingestion/schema.sql") -> None:
    """Create tables if they don't exist by executing schema.sql."""
    engine = get_engine()
    with open(schema_path, "r", encoding="utf-8") as fh:
        ddl = fh.read()
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Database schema initialised.")


# ── Generic helpers ───────────────────────────────────────────────
def read_table(table: str, where: str | None = None) -> pd.DataFrame:
    """Read a full table (or filtered subset) into a DataFrame."""
    query = f"SELECT * FROM {table}"
    if where:
        query += f" WHERE {where}"
    engine = get_engine()
    df = pd.read_sql(query, engine)
    logger.info(f"Read {len(df):,} rows from {table}")
    return df


def read_query(sql: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(sql, engine)


# ── Upsert helpers ────────────────────────────────────────────────
_BOOL_COLS = ["partner", "dependents", "phone_service", "paperless_billing"]
_DATE_COLS = ["signup_date", "churn_date"]


def _normalise_customers(df: pd.DataFrame) -> pd.DataFrame:
    """CSV text -> proper types (Yes/No -> bool, strings -> dates) so INSERT..SELECT casts work."""
    df = df.copy()
    for c in _BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().isin({"yes", "true", "1", "y", "t"})
    for c in _DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def upsert_customers(df: pd.DataFrame) -> None:
    """Upsert customer records.  Conflict key: customer_id."""
    if df.empty:
        return
    df = _normalise_customers(df)
    engine = get_engine()
    tmp = "tmp_customers"
    cols = ", ".join(df.columns)
    with engine.begin() as conn:
        df.to_sql(tmp, conn, if_exists="replace", index=False, method="multi", chunksize=500)
        conn.execute(text(f"""
            INSERT INTO customers ({cols})
                SELECT {cols} FROM {tmp}
            ON CONFLICT (customer_id) DO UPDATE
                SET updated_at      = NOW(),
                    churn_date      = EXCLUDED.churn_date,
                    monthly_charges = EXCLUDED.monthly_charges,
                    total_charges   = EXCLUDED.total_charges,
                    churn_label     = EXCLUDED.churn_label;
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {tmp};"))
    logger.info(f"Upserted {len(df):,} customers.")


def upsert_tickets(df: pd.DataFrame) -> None:
    """
    Upsert support tickets.  Conflict key: ticket_id (if pre-assigned).
    For synthetic data with serial IDs, we truncate-and-reload.
    """
    if df.empty:
        return
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE tickets RESTART IDENTITY CASCADE;"))
        df.to_sql("tickets", conn, if_exists="append", index=False, method="multi", chunksize=500)
    logger.info(f"Loaded {len(df):,} tickets.")


def upsert_events(df: pd.DataFrame) -> None:
    """Load behavioural events (truncate-and-reload for daily batch)."""
    if df.empty:
        return
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE events RESTART IDENTITY;"))
        df.to_sql("events", conn, if_exists="append", index=False, method="multi", chunksize=1000)
    logger.info(f"Loaded {len(df):,} events.")


def upsert_features(df: pd.DataFrame) -> None:
    """Upsert feature rows.  Conflict key: (customer_id, snapshot_date)."""
    if df.empty:
        return
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    engine = get_engine()
    tmp = "tmp_features"
    cols = list(df.columns)
    col_list = ", ".join(cols)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("customer_id", "snapshot_date"))
    with engine.begin() as conn:
        df.to_sql(tmp, conn, if_exists="replace", index=False, method="multi", chunksize=500)
        conn.execute(text(f"""
            INSERT INTO features ({col_list})
                SELECT {col_list} FROM {tmp}
            ON CONFLICT (customer_id, snapshot_date) DO UPDATE SET {updates};
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {tmp};"))
    logger.info(f"Upserted {len(df):,} feature rows.")


def write_predictions(df: pd.DataFrame) -> None:
    """Append batch prediction rows (each run appends a new dated batch)."""
    if df.empty:
        return
    engine = get_engine()
    with engine.begin() as conn:
        df.to_sql("predictions", conn, if_exists="append", index=False, method="multi", chunksize=500)
    logger.info(f"Wrote {len(df):,} predictions.")
