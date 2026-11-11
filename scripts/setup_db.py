"""
scripts/setup_db.py
────────────────────
Initialise the database schema and load generated CSV files
from data/raw/ into PostgreSQL.

Run AFTER:
  1. docker compose up -d postgres
  2. python scripts/generate_data.py

Usage:
  python scripts/setup_db.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.db_loader import (
    init_schema,
    upsert_customers,
    upsert_events,
    upsert_tickets,
)

SNAPSHOT_DATE = os.environ.get("SNAPSHOT_DATE", "2026-11-01")
RAW_DIR       = Path("data/raw")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error(f"File not found: {path}\nRun scripts/generate_data.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    logger.info(f"Read {len(df):,} rows from {path}")
    return df


def main() -> None:
    logger.info("=== setup_db.py ===")

    # 1. Initialise schema
    logger.info("Initialising database schema …")
    init_schema("src/ingestion/schema.sql")

    # 2. Load CSVs
    crm_path     = RAW_DIR / "crm"     / SNAPSHOT_DATE / "crm.csv"
    events_path  = RAW_DIR / "logs"    / SNAPSHOT_DATE / "events.csv"
    tickets_path = RAW_DIR / "tickets" / SNAPSHOT_DATE / "tickets.csv"

    crm     = load_csv(crm_path)
    events  = load_csv(events_path)
    tickets = load_csv(tickets_path)

    # 3. Upsert into PostgreSQL
    logger.info("Loading customers …")
    upsert_customers(crm)

    logger.info("Loading events …")
    upsert_events(events)

    logger.info("Loading tickets …")
    upsert_tickets(tickets)

    logger.info(
        "\n"
        "─────────────────────────────────────────\n"
        "  Database setup complete.\n"
        f"  Customers : {len(crm):>6,}\n"
        f"  Events    : {len(events):>6,}\n"
        f"  Tickets   : {len(tickets):>6,}\n"
        "─────────────────────────────────────────\n"
        "Next step: docker compose up -d"
    )


if __name__ == "__main__":
    main()
