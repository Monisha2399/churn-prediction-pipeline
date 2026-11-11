"""
src/ingestion/s3_loader.py
──────────────────────────
Load raw CSV files from S3 into local DataFrames or stream directly
into PostgreSQL.  Uses boto3; credentials come from the EC2 IAM role
(no long-lived keys in code or environment).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


# ── Config ────────────────────────────────────────────────────────
BUCKET: str = os.environ.get("S3_BUCKET", "churn-bucket")
REGION: str = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")


def _s3_client():
    """Return a boto3 S3 client.  IAM role is preferred over explicit keys."""
    return boto3.client("s3", region_name=REGION)


# ── Helpers ───────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _download_to_buffer(bucket: str, key: str) -> bytes:
    """Download an S3 object, retrying on transient errors."""
    client = _s3_client()
    logger.info(f"Downloading s3://{bucket}/{key}")
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def load_csv_from_s3(
    s3_key: str,
    bucket: str = BUCKET,
    **pandas_kwargs,
) -> pd.DataFrame:
    """
    Read a CSV file from S3 and return it as a DataFrame.

    Parameters
    ----------
    s3_key : str
        Object key inside the bucket, e.g. ``raw/crm/2026-11-04/crm.csv``.
    bucket : str
        S3 bucket name (defaults to env var ``S3_BUCKET``).
    **pandas_kwargs
        Extra kwargs forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    raw = _download_to_buffer(bucket, s3_key)
    df = pd.read_csv(io.BytesIO(raw), **pandas_kwargs)
    logger.info(f"Loaded {len(df):,} rows from s3://{bucket}/{s3_key}")
    return df


def load_parquet_from_s3(
    s3_key: str,
    bucket: str = BUCKET,
) -> pd.DataFrame:
    """Read a Parquet file from S3."""
    raw = _download_to_buffer(bucket, s3_key)
    df = pd.read_parquet(io.BytesIO(raw))
    logger.info(f"Loaded {len(df):,} rows (parquet) from s3://{bucket}/{s3_key}")
    return df


def upload_dataframe_to_s3(
    df: pd.DataFrame,
    s3_key: str,
    bucket: str = BUCKET,
    fmt: str = "parquet",
) -> None:
    """
    Upload a DataFrame to S3 as Parquet (default) or CSV.

    Parameters
    ----------
    df : pd.DataFrame
    s3_key : str
    bucket : str
    fmt : str
        ``"parquet"`` or ``"csv"``.
    """
    client = _s3_client()
    buf = io.BytesIO()

    if fmt == "parquet":
        df.to_parquet(buf, index=False, engine="pyarrow")
        content_type = "application/octet-stream"
    elif fmt == "csv":
        df.to_csv(buf, index=False)
        content_type = "text/csv"
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    buf.seek(0)
    client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=buf.getvalue(),
        ContentType=content_type,
    )
    logger.info(f"Uploaded {len(df):,} rows to s3://{bucket}/{s3_key} [{fmt}]")


def list_s3_keys(prefix: str, bucket: str = BUCKET) -> list[str]:
    """List all object keys under a given S3 prefix."""
    client = _s3_client()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    logger.info(f"Found {len(keys)} keys under s3://{bucket}/{prefix}")
    return keys


def key_exists(s3_key: str, bucket: str = BUCKET) -> bool:
    """Return True if the S3 object exists (used for cache checks)."""
    client = _s3_client()
    try:
        client.head_object(Bucket=bucket, Key=s3_key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            return False
        raise
