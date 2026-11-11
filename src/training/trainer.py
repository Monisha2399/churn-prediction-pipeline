"""
src/training/trainer.py
────────────────────────
Trains four experiments as described in the project guide:

  Exp 1 — Baseline:      CRM + behavioural  →  Logistic Regression
  Exp 2 — + Classic NLP: + TF-IDF + Sentiment  →  Logistic Regression
  Exp 3 — + BERT:        + BERT embeddings  →  Logistic Regression
  Exp 4 — Best effort:   All features  →  HistGradientBoostingClassifier

Each experiment is logged as a separate MLflow run.
Uses sklearn Pipeline so the same transforms are applied at serving time.
class_weight="balanced" to handle churn imbalance (~10-25%).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.evaluate import evaluate_model, log_evaluation_to_mlflow

# ── MLflow config ─────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "churn_prediction"

# ── Feature column groups ─────────────────────────────────────────
CRM_COLS = [
    "tenure_months", "monthly_charges", "contract_encoded", "senior_citizen",
]
BEHAV_COLS = [
    "logins_30d", "session_min_30d", "login_trend",
    "failed_pay_30d", "days_since_login",
]
TICKET_COLS = [
    "ticket_count_30d", "ticket_count_60d", "ticket_count_90d",
    "open_tickets", "escalations_30d", "days_since_ticket",
]
SENTIMENT_COLS = ["sentiment_mean", "sentiment_min", "sentiment_trend"]

STRUCTURED_COLS = CRM_COLS + BEHAV_COLS + TICKET_COLS + SENTIMENT_COLS


def _get_tfidf_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("tfidf_")]


def _get_bert_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("emb_")]


# Runs whose inputs are exactly the columns the API / batch scorer can supply
SERVING_RUNS = ("baseline_lr", "structured_hgbc")


def _select_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Select only columns that exist in df (graceful degradation, no duplicates)."""
    return df[[c for c in dict.fromkeys(cols) if c in df.columns]]


# ── MLflow setup ──────────────────────────────────────────────────
def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI} | experiment: {EXPERIMENT_NAME}")


# ── Single experiment runner ──────────────────────────────────────
def train_experiment(
    run_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model,
    params: Dict[str, Any],
    threshold: float = 0.4,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train one model, evaluate, and log everything to MLflow.

    Returns
    -------
    (fitted_pipeline, metrics_dict)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    model),
    ])

    with mlflow.start_run(run_name=run_name):
        # Log params
        mlflow.log_params(params)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test",  len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("threshold", threshold)

        logger.info(f"Training [{run_name}] — {X_train.shape[1]} features …")
        pipeline.fit(X_train, y_train)

        # Evaluate
        metrics, fig = evaluate_model(pipeline, X_test, y_test, threshold=threshold)
        log_evaluation_to_mlflow(metrics, fig)

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=None,   # registration handled separately
        )

        logger.info(
            f"[{run_name}] ROC-AUC={metrics['roc_auc']:.4f}  "
            f"F1={metrics['f1']:.4f}  Recall={metrics['recall']:.4f}"
        )

    return pipeline, metrics


# ── Four experiment suite ─────────────────────────────────────────
def run_all_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 0.4,
) -> Dict[str, Dict[str, float]]:
    """
    Run all four experiments and return metrics keyed by experiment name.

    Parameters
    ----------
    train_df  : feature table (train split) including label column
    test_df   : feature table (test split) including label column
    threshold : classification threshold (tune on validation set)

    Returns
    -------
    dict: {experiment_name: metrics_dict}
    """
    setup_mlflow()
    results: Dict[str, Dict[str, float]] = {}

    y_train = train_df["label"]
    y_test  = test_df["label"]

    # ── Experiment 1: Baseline ────────────────────────────────────
    base_cols = STRUCTURED_COLS
    X_tr1 = _select_cols(train_df, base_cols).astype(float)
    X_te1 = _select_cols(test_df,  base_cols).astype(float)

    _, m1 = train_experiment(
        run_name  = "baseline_lr",
        X_train   = X_tr1,
        y_train   = y_train,
        X_test    = X_te1,
        y_test    = y_test,
        model     = LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0, solver="lbfgs"
        ),
        params    = {"model": "LogisticRegression", "features": "CRM+Behavioural",
                     "C": 1.0, "class_weight": "balanced"},
        threshold = threshold,
    )
    results["baseline_lr"] = m1

    # ── Experiment 2: + Classic NLP ───────────────────────────────
    tfidf_cols = _get_tfidf_cols(train_df)
    nlp_cols   = base_cols + tfidf_cols
    X_tr2 = _select_cols(train_df, nlp_cols).astype(float)
    X_te2 = _select_cols(test_df,  nlp_cols).astype(float)

    _, m2 = train_experiment(
        run_name  = "classic_nlp_lr",
        X_train   = X_tr2,
        y_train   = y_train,
        X_test    = X_te2,
        y_test    = y_test,
        model     = LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0, solver="saga"
        ),
        params    = {"model": "LogisticRegression", "features": "CRM+Behavioural+TFIDF+Sentiment",
                     "C": 1.0, "class_weight": "balanced"},
        threshold = threshold,
    )
    results["classic_nlp_lr"] = m2

    # ── Experiment 3: + BERT ──────────────────────────────────────
    bert_cols   = _get_bert_cols(train_df)
    bert_feature_cols = STRUCTURED_COLS + SENTIMENT_COLS + bert_cols
    X_tr3 = _select_cols(train_df, bert_feature_cols).astype(float)
    X_te3 = _select_cols(test_df,  bert_feature_cols).astype(float)

    _, m3 = train_experiment(
        run_name  = "bert_lr",
        X_train   = X_tr3,
        y_train   = y_train,
        X_test    = X_te3,
        y_test    = y_test,
        model     = LogisticRegression(
            class_weight="balanced", max_iter=1000, C=0.5, solver="saga"
        ),
        params    = {"model": "LogisticRegression", "features": "CRM+Behavioural+Sentiment+BERT",
                     "C": 0.5, "class_weight": "balanced"},
        threshold = threshold,
    )
    results["bert_lr"] = m3

    # ── Experiment 4: Best effort — all features ──────────────────
    all_cols = STRUCTURED_COLS + tfidf_cols + bert_cols
    X_tr4 = _select_cols(train_df, all_cols).astype(float)
    X_te4 = _select_cols(test_df,  all_cols).astype(float)

    _, m4 = train_experiment(
        run_name  = "best_effort_hgbc",
        X_train   = X_tr4,
        y_train   = y_train,
        X_test    = X_te4,
        y_test    = y_test,
        model     = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=int(os.environ.get("RANDOM_SEED", 42)),
        ),
        params    = {"model": "HistGradientBoostingClassifier", "features": "All",
                     "max_iter": 300, "learning_rate": 0.05, "max_depth": 6},
        threshold = threshold,
    )
    results["best_effort_hgbc"] = m4

    # ── Experiment 5: structured-only gradient boosting (servable via the API) ──
    X_tr5 = _select_cols(train_df, base_cols).astype(float)
    X_te5 = _select_cols(test_df,  base_cols).astype(float)
    _, m5 = train_experiment(
        run_name  = "structured_hgbc",
        X_train   = X_tr5, y_train = y_train, X_test = X_te5, y_test = y_test,
        model     = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=40,
            class_weight="balanced", random_state=int(os.environ.get("RANDOM_SEED", 42)),
        ),
        params    = {"model": "HistGradientBoostingClassifier", "features": "CRM+Behavioural+Ticket+Sentiment",
                     "max_iter": 200, "learning_rate": 0.05, "max_depth": 4},
        threshold = threshold,
    )
    results["structured_hgbc"] = m5

    # ── Print comparison table ────────────────────────────────────
    logger.info("\n" + "="*65)
    logger.info(f"{'Experiment':<25} {'ROC-AUC':>8} {'F1':>7} {'Recall':>8} {'Precision':>10}")
    logger.info("="*65)
    for name, m in results.items():
        logger.info(
            f"{name:<25} {m['roc_auc']:>8.4f} {m['f1']:>7.4f} "
            f"{m['recall']:>8.4f} {m['precision']:>10.4f}"
        )
    logger.info("="*65)

    return results


# ── Champion promotion ────────────────────────────────────────────
def promote_champion(
    results: Dict[str, Dict[str, float]],
    metric: str = "roc_auc",
    model_name: str = "churn_classifier",
) -> str:
    """
    Register the best experiment as 'champion' in the MLflow Model Registry.

    Returns
    -------
    The run name of the promoted champion.
    """
    best_run = max(results, key=lambda k: results[k].get(metric, 0))
    best_score = results[best_run][metric]
    logger.info(f"Champion: {best_run} ({metric}={best_score:.4f})")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{best_run}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        logger.warning(f"Could not find run '{best_run}' to promote.")
        return best_run

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    # Register and alias as champion
    mv = mlflow.register_model(model_uri, model_name)
    client.set_registered_model_alias(model_name, "champion", mv.version)
    logger.info(f"Registered '{model_name}' v{mv.version} as @champion (run={run_id[:8]})")

    return best_run
