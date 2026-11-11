"""
src/pipeline_steps.py
──────────────────────
One CLI entry point per pipeline stage. Airflow runs each as its own process
(inside the /opt/venv environment) so the Airflow image and the ML stack never
share a Python environment.

    python -m src.pipeline_steps <ingest|preprocess|features|train|promote|score>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from loguru import logger

DATA_SNAPSHOT = os.environ.get("DATA_SNAPSHOT", "2026-11-01")          # S3 folder of raw data
SNAPSHOTS = [s.strip() for s in os.environ.get(
    "FEATURE_SNAPSHOTS", "2026-08-01,2026-09-01,2026-10-01,2026-11-01").split(",")]
ART_DIR = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
FEATURES_KEY = "features/feature_table.parquet"
RESULTS_FILE = ART_DIR / "results.json"
THRESHOLD = float(os.environ.get("CHURN_THRESHOLD", "0.4"))

DB_COLS = [
    "customer_id", "snapshot_date", "label",
    "tenure_months", "monthly_charges", "contract_encoded", "senior_citizen",
    "logins_30d", "session_min_30d", "login_trend", "failed_pay_30d", "days_since_login",
    "ticket_count_30d", "ticket_count_60d", "ticket_count_90d",
    "open_tickets", "escalations_30d", "days_since_ticket",
    "sentiment_mean", "sentiment_min", "sentiment_trend",
]


def ingest() -> None:
    from src.ingestion.db_loader import upsert_customers, upsert_events, upsert_tickets
    from src.ingestion.s3_loader import load_csv_from_s3
    bucket = os.environ["S3_BUCKET"]
    crm = load_csv_from_s3(f"raw/crm/{DATA_SNAPSHOT}/crm.csv", bucket=bucket)
    tickets = load_csv_from_s3(f"raw/tickets/{DATA_SNAPSHOT}/tickets.csv", bucket=bucket)
    events = load_csv_from_s3(f"raw/logs/{DATA_SNAPSHOT}/events.csv", bucket=bucket)
    upsert_customers(crm)        # parents first (foreign keys)
    upsert_tickets(tickets)
    upsert_events(events)


def preprocess() -> None:
    """Data-quality gate + cleaning smoke test (features step re-runs the cleaners)."""
    from src.ingestion.db_loader import read_query, read_table
    from src.preprocessing.cleaner import clean_ticket_df
    counts = read_query("SELECT (SELECT COUNT(*) FROM customers) c, (SELECT COUNT(*) FROM tickets) t, "
                        "(SELECT COUNT(*) FROM events) e").iloc[0]
    logger.info(f"Row counts: {counts.to_dict()}")
    if min(counts["c"], counts["t"], counts["e"]) == 0:
        raise RuntimeError("An input table is empty - run ingest first.")
    clean = clean_ticket_df(read_table("tickets"))
    if clean["body_bert"].str.len().eq(0).mean() > 0.05:
        raise RuntimeError("More than 5% of tickets are empty after cleaning.")


def features() -> None:
    from src.features.feature_store import build_feature_table
    from src.features.tfidf_features import TFIDFFeatureBuilder
    from src.ingestion.db_loader import read_table, upsert_features
    from src.ingestion.s3_loader import upload_dataframe_to_s3
    from src.preprocessing.cleaner import clean_ticket_df

    customers, tickets, events = read_table("customers"), read_table("tickets"), read_table("events")
    snaps = [pd.Timestamp(s) for s in SNAPSHOTS]
    if len(snaps) < 3:
        raise RuntimeError("Need >= 3 snapshots (train, validation-ish, test).")

    # TF-IDF vocabulary is fitted ONLY on tickets older than the last training snapshot (no test-period text).
    tc = clean_ticket_df(tickets)
    fit_text = tc.loc[tc["created_at"] < pd.Timestamp(snaps[-2], tz="UTC"), "body_tfidf"]
    tfidf = TFIDFFeatureBuilder(max_features=int(os.environ.get("TFIDF_MAX_FEATURES", "1000"))).fit(fit_text)

    tables = [build_feature_table(customers, tickets, events, snapshot_date=s,
                                  tfidf_builder=tfidf, include_bert=True, include_tfidf=True)
              for s in snaps]
    feat = pd.concat(tables, ignore_index=True)
    feat["snapshot_date"] = pd.to_datetime(feat["snapshot_date"])
    num = feat.select_dtypes("number").columns.difference(["label"])
    feat[num] = feat[num].fillna(0)
    heavy = [c for c in feat.columns if c.startswith(("tfidf_", "emb_"))]
    feat[heavy] = feat[heavy].astype("float32")
    logger.info(f"Feature table: {feat.shape}, positives per snapshot:\n"
                f"{feat.groupby('snapshot_date')['label'].agg(['size', 'mean'])}")

    upload_dataframe_to_s3(feat, FEATURES_KEY, bucket=os.environ["S3_BUCKET"], fmt="parquet")

    db = feat[[c for c in DB_COLS if c in feat.columns]].copy()
    for c in ("login_trend", "sentiment_trend"):          # fit NUMERIC(8,4)
        db[c] = db[c].clip(-9999, 9999)
    upsert_features(db)


def train() -> None:
    from src.features.feature_store import time_based_split
    from src.ingestion.s3_loader import load_parquet_from_s3
    from src.training.trainer import run_all_experiments
    feat = load_parquet_from_s3(FEATURES_KEY, bucket=os.environ["S3_BUCKET"])
    n_snap = feat["snapshot_date"].nunique()
    train_df, test_df = time_based_split([feat], test_fraction=1.0 / n_snap)   # last snapshot = test
    results = run_all_experiments(train_df, test_df, threshold=THRESHOLD)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, default=float, indent=2))


def promote() -> None:
    import mlflow
    from src.training.trainer import SERVING_RUNS, promote_champion
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    results = json.loads(RESULTS_FILE.read_text())
    serving = {k: v for k, v in results.items() if k in SERVING_RUNS}      # API can only serve these
    best = max(serving, key=lambda k: serving[k]["roc_auc"])
    new_auc = serving[best]["roc_auc"]
    client = mlflow.tracking.MlflowClient()
    name = os.environ.get("CHAMPION_MODEL_NAME", "churn_classifier")
    try:
        mv = client.get_model_version_by_alias(name, "champion")
        champ_auc = float(client.get_run(mv.run_id).data.metrics.get("roc_auc", 0))
    except Exception:
        champ_auc = 0.0
    if new_auc > champ_auc + 0.001:
        promote_champion(serving, model_name=name)
        logger.info(f"Promoted {best}: ROC-AUC {new_auc:.4f} > champion {champ_auc:.4f}")
    else:
        logger.info(f"Kept champion ({champ_auc:.4f}); best new = {best} {new_auc:.4f}")


def score() -> None:
    from sqlalchemy import text
    from src.api.model_loader import get_model_version, load_champion_model, predict_batch
    from src.ingestion.db_loader import get_engine, write_predictions
    from src.ingestion.s3_loader import load_parquet_from_s3
    load_champion_model()
    feat = load_parquet_from_s3(FEATURES_KEY, bucket=os.environ["S3_BUCKET"])
    latest = feat["snapshot_date"].max()
    cur = feat[feat["snapshot_date"] == latest].reset_index(drop=True)      # active customers at latest T
    res = predict_batch(cur.drop(columns=["label", "snapshot_date"]).to_dict("records"), threshold=THRESHOLD)
    preds = pd.DataFrame({
        "customer_id": cur["customer_id"], "snapshot_date": latest.date(),
        "churn_probability": [r[0] for r in res], "churn_label": [r[1] for r in res],
        "model_version": get_model_version(),
    })
    with get_engine().begin() as conn:                                      # idempotent re-run
        conn.execute(text("DELETE FROM predictions WHERE snapshot_date = :d AND model_version = :v"),
                     {"d": latest.date(), "v": get_model_version()})
    write_predictions(preds)
    logger.info(f"Scored {len(preds):,} customers; predicted churners: {int(preds['churn_label'].sum()):,}")


STEPS = {"ingest": ingest, "preprocess": preprocess, "features": features,
         "train": train, "promote": promote, "score": score}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--ds", default=None, help="Airflow logical date (informational)")
    a = ap.parse_args()
    logger.info(f"=== step={a.step} ds={a.ds} data_snapshot={DATA_SNAPSHOT} ===")
    STEPS[a.step]()
