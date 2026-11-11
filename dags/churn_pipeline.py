"""
Airflow DAG: churn_pipeline  (weekly, Mondays 02:00 UTC)

ingest_from_s3 → preprocess → build_features → train_and_log → evaluate_and_promote → batch_score

Each task runs `python -m src.pipeline_steps <step>` in the dedicated ML virtualenv (/opt/venv),
so Airflow's own dependencies never clash with pandas / SQLAlchemy 2 / torch / mlflow.
Every step is idempotent. Credentials come from container env vars / the EC2 IAM role.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PY = "/opt/venv/bin/python"


def step(task_id: str, name: str) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd /opt/airflow && {PY} -m src.pipeline_steps {name} --ds {{{{ ds }}}}",
    )


with DAG(
    dag_id="churn_pipeline",
    description="End-to-end customer churn prediction pipeline",
    schedule="0 2 * * 1",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"owner": "churn-team", "retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["churn", "mlops"],
) as dag:
    t1 = step("ingest_from_s3", "ingest")
    t2 = step("preprocess", "preprocess")
    t3 = step("build_features", "features")
    t4 = step("train_and_log", "train")
    t5 = step("evaluate_and_promote", "promote")
    t6 = step("batch_score", "score")
    t1 >> t2 >> t3 >> t4 >> t5 >> t6
