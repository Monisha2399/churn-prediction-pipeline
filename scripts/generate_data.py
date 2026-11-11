"""
scripts/generate_data.py
─────────────────────────
Generates the full synthetic dataset for the churn prediction pipeline.

IMPORTANT — Kaggle Data Required:
──────────────────────────────────
This script does NOT create CRM customer records from scratch.
It expects the IBM Telco Customer Churn dataset downloaded from Kaggle:

    https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download steps (see README for full guide):
  1. Go to the URL above and click "Download"
  2. Place the file at:  data/raw/crm/WA_Fn-UseC_-Telco-Customer-Churn.csv
  3. Then run:  python scripts/generate_data.py

What this script generates ON TOP of the Kaggle CRM data:
  - signup_date and churn_date columns
  - Synthetic behavioural logs (daily: logins, session_minutes, etc.)
  - Synthetic support tickets (free-text, category, priority)

All generated data is saved under data/raw/ and also uploaded to S3
if the S3_BUCKET environment variable is set.

Random seed: controlled by RANDOM_SEED env var (default 42).
"""

from __future__ import annotations

import os
import random
import sys
import textwrap
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── Config ────────────────────────────────────────────────────────
RANDOM_SEED    = int(os.environ.get("RANDOM_SEED", 42))
OUTPUT_DIR     = Path("data/raw")
SAMPLE_DIR     = Path("data/sample")
CRM_INPUT_PATH = Path("data/raw/crm/WA_Fn-UseC_-Telco-Customer-Churn.csv")
SNAPSHOT_DATE  = date(2026, 11, 1)      # Reference snapshot
HISTORY_DAYS   = 200                    # How far back to generate logs

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ── Helpers ───────────────────────────────────────────────────────
def _rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=int(rng.integers(0, delta)))


# ── Step 1: Load and enrich CRM from Kaggle ───────────────────────
def load_and_enrich_crm() -> pd.DataFrame:
    """
    Load IBM Telco CSV and add signup_date / churn_date columns
    needed for the 30-day label window.
    """
    if not CRM_INPUT_PATH.exists():
        logger.error(
            f"\n{'='*60}\n"
            f"  CRM file not found at: {CRM_INPUT_PATH}\n\n"
            f"  Download the IBM Telco Customer Churn dataset from Kaggle:\n"
            f"  https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n\n"
            f"  Place the CSV at:\n"
            f"  {CRM_INPUT_PATH.resolve()}\n\n"
            f"  Then re-run this script.\n"
            f"{'='*60}"
        )
        sys.exit(1)

    df = pd.read_csv(CRM_INPUT_PATH)
    logger.info(f"Loaded {len(df):,} customers from Kaggle CRM file.")

    # Rename to snake_case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename customerid → customer_id
    if "customerid" in df.columns:
        df = df.rename(columns={"customerid": "customer_id"})

    # Convert tenure to numeric
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    df = df.rename(columns={"tenure": "tenure_months"})

    # Generate signup_date (SNAPSHOT_DATE - tenure_months)
    df["signup_date"] = df["tenure_months"].apply(
        lambda t: SNAPSHOT_DATE - timedelta(days=int(t * 30.44))
    )

    # Generate churn_date for churned customers
    # churn_date is set within the 30-day window after snapshot for a random subset
    # and in the past for historical churners
    df["churn_label_orig"] = (df["churn"].str.lower() == "yes").astype(int)

    # Assign churn_date:
    #   - 40% of churners: churn_date in (snapshot, snapshot+30] — these become label=1
    #   - 60% of churners: churn_date before snapshot — historical churners (label=0)
    #   - non-churners: churn_date = None
    churn_mask = df["churn_label_orig"] == 1
    # Every churner gets a churn_date spread over the 120 days before the snapshot and
    # the 30 days after it, so each monthly snapshot (Aug-Nov) has positive labels.
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    offsets = rng.integers(-120, 31, size=int(churn_mask.sum()))
    churn = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    churn[churn_mask] = pd.Timestamp(SNAPSHOT_DATE) + pd.to_timedelta(offsets, unit="D")
    # nobody churns within 14 days of signing up
    earliest = df["signup_date"] + pd.Timedelta(days=14)
    df["churn_date"] = churn.where(churn.isna() | (churn >= earliest), earliest)

    # Map Kaggle columns to schema
    col_map = {
        "gender":           "gender",
        "seniorcitizen":    "senior_citizen",
        "partner":          "partner",
        "dependents":       "dependents",
        "tenure_months":    "tenure_months",
        "phoneservice":     "phone_service",
        "multiplelines":    "multiple_lines",
        "internetservice":  "internet_service",
        "onlinesecurity":   "online_security",
        "onlinebackup":     "online_backup",
        "deviceprotection": "device_protection",
        "techsupport":      "tech_support",
        "streamingtv":      "streaming_tv",
        "streamingmovies":  "streaming_movies",
        "contract":         "contract_type",
        "paperlessbilling": "paperless_billing",
        "paymentmethod":    "payment_method",
        "monthlycharges":   "monthly_charges",
        "totalcharges":     "total_charges",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    df["total_charges"] = pd.to_numeric(df.get("total_charges", 0), errors="coerce").fillna(0)
    df["churn_label"]   = df["churn_label_orig"]

    keep = [
        "customer_id", "signup_date", "churn_date", "gender", "senior_citizen",
        "partner", "dependents", "tenure_months", "phone_service", "multiple_lines",
        "internet_service", "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies", "contract_type",
        "paperless_billing", "payment_method", "monthly_charges", "total_charges",
        "churn_label",
    ]
    df = df[[c for c in keep if c in df.columns]]
    logger.info(
        f"CRM enriched: {len(df):,} customers | "
        f"label=1: {df['churn_label'].sum():,} ({df['churn_label'].mean():.1%})"
    )
    return df


# ── Step 2: Behavioural logs ──────────────────────────────────────
TICKET_TEMPLATES_COMPLAINT = [
    "My service has been down for {n} days and no one is helping.",
    "I have been overcharged {n} times this month.",
    "Terrible customer support, been waiting {n} hours.",
    "Internet keeps dropping every {n} minutes, this is unacceptable.",
    "Still waiting for refund after {n} days. Very disappointed.",
    "Cannot access my account for {n} days. Extremely frustrated.",
    "Speed is way below what I'm paying for. Getting only {n} Mbps.",
]

TICKET_TEMPLATES_NEUTRAL = [
    "How do I update my billing address?",
    "Can I change my plan from monthly to annual?",
    "I would like to add international calling to my plan.",
    "Please confirm my next billing date.",
    "How do I reset my router?",
    "What is included in the {n}GB data plan?",
    "Can I get a copy of my last {n} invoices?",
]

TICKET_CATEGORIES = ["billing", "technical", "cancellation", "general"]
TICKET_PRIORITIES = ["low", "medium", "high", "critical"]
TICKET_STATUSES   = ["open", "resolved", "escalated"]


def generate_events(customers: pd.DataFrame) -> pd.DataFrame:
    """Daily behavioural logs. Usage is tied to each customer's OWN churn_date:
    - 70% of churners decline in the 45 days before leaving (30% leave silently)
    - 12% of everyone has a random 21-day dip (loyal customers dip too)
    - no events after a customer's churn_date
    """
    start = pd.Timestamp(SNAPSHOT_DATE) - pd.Timedelta(days=HISTORY_DAYS)
    end   = pd.Timestamp(SNAPSHOT_DATE) - pd.Timedelta(days=1)
    frames = []
    for cid, signup, churn in zip(customers["customer_id"],
                                  pd.to_datetime(customers["signup_date"]),
                                  pd.to_datetime(customers["churn_date"])):
        first = max(start, signup)
        last  = end if pd.isna(churn) else min(end, churn)
        if last < first:
            continue
        days = pd.date_range(first, last, freq="D")
        n = len(days)
        base  = rng.gamma(4.0, 0.5)               # per-customer activity level (mean 2)
        decay = np.ones(n)
        near  = np.zeros(n, dtype=bool)
        if pd.notna(churn):
            d2c  = (churn - days).days.to_numpy()
            near = d2c < 45
            if rng.random() < 0.7:
                strength = rng.uniform(0.3, 0.8)
                decay = 1 - strength * np.clip((45 - d2c) / 45, 0, 1)
        if n > 30 and rng.random() < 0.12:
            s = int(rng.integers(0, n - 21))
            decay[s:s + 21] *= 0.5
        frames.append(pd.DataFrame({
            "customer_id":     cid,
            "event_date":      days.strftime("%Y-%m-%d"),
            "logins":          rng.poisson(base * decay),
            "session_minutes": np.round(rng.exponential(25.0, n) * decay * base / 2.0, 2),
            "features_used":   rng.poisson(3.0 * decay),
            "failed_payments": rng.binomial(1, np.where(near, 0.10, 0.05)),
            "support_calls":   rng.binomial(1, np.where(near, 0.07, 0.03)),
        }))
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Generated {len(df):,} event rows for {len(customers):,} customers.")
    return df


def generate_tickets(customers: pd.DataFrame) -> pd.DataFrame:
    """Support tickets. 60% of a (non-calm) churner's tickets fall in the 60 days before
    churn. 30% of churners are 'calm leavers' and 10% of stayers are 'angry stayers'."""
    start = pd.Timestamp(SNAPSHOT_DATE) - pd.Timedelta(days=HISTORY_DAYS)
    end   = pd.Timestamp(SNAPSHOT_DATE) - pd.Timedelta(days=1)
    rows = []
    for cid, signup, churn in zip(customers["customer_id"],
                                  pd.to_datetime(customers["signup_date"]),
                                  pd.to_datetime(customers["churn_date"])):
        first = max(start, signup)
        last  = end if pd.isna(churn) else min(end, churn - pd.Timedelta(days=1))
        if last < first:
            continue
        churner     = pd.notna(churn)
        calm_leaver = churner and rng.random() < 0.30
        angry_stay  = (not churner) and rng.random() < 0.10
        hot = (churner and not calm_leaver) or angry_stay
        n_tickets = min(int(rng.negative_binomial(2, 0.40 if hot else 0.65)), 10)

        for _ in range(n_tickets):
            lo = first
            if churner and not calm_leaver and rng.random() < 0.6:
                lo = max(first, churn - pd.Timedelta(days=60))
                if lo > last:
                    lo = first
            day = lo + pd.Timedelta(days=int(rng.integers(0, (last - lo).days + 1)))
            is_complaint = rng.random() < (0.55 if hot else 0.15)
            templates = TICKET_TEMPLATES_COMPLAINT if is_complaint else TICKET_TEMPLATES_NEUTRAL
            body = random.choice(templates).format(n=random.randint(1, 10))
            category = "cancellation" if (is_complaint and rng.random() < 0.2) else random.choice(TICKET_CATEGORIES)
            priority = random.choices(TICKET_PRIORITIES,
                weights=[0.1, 0.25, 0.4, 0.25] if is_complaint else [0.4, 0.3, 0.2, 0.1])[0]
            status = random.choices(TICKET_STATUSES,
                weights=[0.3, 0.5, 0.2] if is_complaint else [0.2, 0.65, 0.15])[0]
            resolved = None
            if status == "resolved":
                resolved = min(end, day + pd.Timedelta(days=random.randint(1, 7))).date().isoformat()
            rows.append({
                "customer_id": cid,
                "created_at":  day.date().isoformat() + "T" + f"{random.randint(8, 20):02d}:00:00",
                "resolved_at": resolved,
                "category":    category,
                "priority":    priority,
                "status":      status,
                "subject":     body[:60],
                "body":        body,
            })
    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df):,} tickets for {len(customers):,} customers.")
    return df


# ── Step 3: Save and upload ───────────────────────────────────────
def save_and_upload(
    crm: pd.DataFrame,
    events: pd.DataFrame,
    tickets: pd.DataFrame,
    snapshot: str,
) -> None:
    """Save CSVs locally and upload to S3 if bucket is configured."""
    crm_dir     = OUTPUT_DIR / "crm"     / snapshot
    events_dir  = OUTPUT_DIR / "logs"    / snapshot
    tickets_dir = OUTPUT_DIR / "tickets" / snapshot
    sample_dir  = SAMPLE_DIR

    for d in [crm_dir, events_dir, tickets_dir, sample_dir]:
        d.mkdir(parents=True, exist_ok=True)

    crm.to_csv(crm_dir / "crm.csv",          index=False)
    events.to_csv(events_dir / "events.csv",  index=False)
    tickets.to_csv(tickets_dir / "tickets.csv", index=False)

    # Save a tiny sample (10 rows each) for repo
    crm.head(10).to_csv(sample_dir / "crm_sample.csv",         index=False)
    events.head(50).to_csv(sample_dir / "events_sample.csv",   index=False)
    tickets.head(20).to_csv(sample_dir / "tickets_sample.csv", index=False)

    logger.info(f"Saved data to {OUTPUT_DIR}")

    # Upload to S3 if configured
    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        try:
            from src.ingestion.s3_loader import upload_dataframe_to_s3
            upload_dataframe_to_s3(crm,     f"raw/crm/{snapshot}/crm.csv",          bucket, "csv")
            upload_dataframe_to_s3(events,  f"raw/logs/{snapshot}/events.csv",      bucket, "csv")
            upload_dataframe_to_s3(tickets, f"raw/tickets/{snapshot}/tickets.csv",  bucket, "csv")
            logger.info(f"Uploaded all raw files to s3://{bucket}/raw/{snapshot}/")
        except Exception as exc:
            logger.warning(f"S3 upload skipped: {exc}")
    else:
        logger.info("S3_BUCKET not set — skipping S3 upload.")


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    snapshot_str = SNAPSHOT_DATE.isoformat()
    logger.info(f"Generating synthetic dataset (snapshot={snapshot_str}, seed={RANDOM_SEED})")

    crm     = load_and_enrich_crm()
    events  = generate_events(crm)
    tickets = generate_tickets(crm)

    save_and_upload(crm, events, tickets, snapshot_str)

    logger.info(
        "\n"
        "─────────────────────────────────────────\n"
        f"  Customers : {len(crm):>6,}\n"
        f"  Events    : {len(events):>6,}\n"
        f"  Tickets   : {len(tickets):>6,}\n"
        f"  Snapshot  : {snapshot_str}\n"
        "─────────────────────────────────────────\n"
        "Next step: python scripts/setup_db.py"
    )


if __name__ == "__main__":
    main()
