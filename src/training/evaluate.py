"""
src/training/evaluate.py
─────────────────────────
Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
confusion matrix and permutation feature importance.
All results are returned as dicts / matplotlib figures for MLflow logging.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless — no display needed on EC2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import mlflow


# ── Core evaluation ───────────────────────────────────────────────
def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.4,
) -> Tuple[Dict[str, float], plt.Figure]:
    """
    Compute classification metrics and confusion matrix.

    Parameters
    ----------
    pipeline  : fitted sklearn Pipeline
    X_test    : test features
    y_test    : true labels
    threshold : decision threshold (tune on validation, not test)

    Returns
    -------
    (metrics_dict, confusion_matrix_figure)
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),                        4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0),       4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0),          4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0),              4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob),                          4),
        "pr_auc":    round(average_precision_score(y_test, y_prob),                4),
        "threshold": threshold,
        "support_positive": int(y_test.sum()),
        "support_total":    int(len(y_test)),
        "churn_rate":       round(y_test.mean(), 4),
    }

    # Confusion matrix figure
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    return metrics, fig


def log_evaluation_to_mlflow(
    metrics: Dict[str, float],
    fig: plt.Figure,
    fig_name: str = "confusion_matrix.png",
) -> None:
    """Log metrics dict and confusion matrix figure to the active MLflow run."""
    mlflow.log_metrics(metrics)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, fig_name)
        fig.savefig(path, dpi=120)
        mlflow.log_artifact(path, artifact_path="plots")

    plt.close(fig)
    logger.debug(f"Logged metrics and {fig_name} to MLflow.")


# ── Feature importance ────────────────────────────────────────────
def compute_permutation_importance(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 10,
    top_n: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation feature importance on a validation set.

    Returns
    -------
    DataFrame with columns: feature, importance_mean, importance_std
    sorted descending by importance_mean (top_n rows).
    """
    logger.info(f"Computing permutation importance (n_repeats={n_repeats}) …")
    result = permutation_importance(
        pipeline, X_val, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature":          X_val.columns.tolist(),
        "importance_mean":  result.importances_mean,
        "importance_std":   result.importances_std,
    }).sort_values("importance_mean", ascending=False).head(top_n)

    return df


def plot_feature_importance(importance_df: pd.DataFrame) -> plt.Figure:
    """Return a horizontal bar chart of top feature importances."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(importance_df) * 0.35)))
    ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance_mean"][::-1],
        xerr=importance_df["importance_std"][::-1],
        color="steelblue",
        alpha=0.85,
    )
    ax.set_xlabel("Permutation Importance (ROC-AUC drop)")
    ax.set_title("Top Feature Importances")
    plt.tight_layout()
    return fig


# ── Champion comparison ───────────────────────────────────────────
def beats_champion(
    new_metrics: Dict[str, float],
    champion_metrics: Dict[str, float],
    metric: str = "roc_auc",
    min_delta: float = 0.001,
) -> bool:
    """
    Return True if new model beats the current champion by at least min_delta.
    Used by Airflow's evaluate_and_promote task.
    """
    new_score  = new_metrics.get(metric, 0)
    champ_score = champion_metrics.get(metric, 0)
    result = (new_score - champ_score) >= min_delta
    logger.info(
        f"Champion comparison ({metric}): "
        f"new={new_score:.4f}  champion={champ_score:.4f}  "
        f"promote={'YES' if result else 'NO'}"
    )
    return result
