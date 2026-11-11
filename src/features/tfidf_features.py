"""
src/features/tfidf_features.py
──────────────────────────────
TF-IDF with bi-grams feature builder.
Fit only on training data; transform on validation/test to prevent leakage.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Constants ─────────────────────────────────────────────────────
MAX_FEATURES: int = 5000
NGRAM_RANGE: tuple = (1, 2)
MIN_DF: int = 3
VECTORIZER_PATH: str = "data/models/tfidf_vectorizer.pkl"


# ── Builder ───────────────────────────────────────────────────────
class TFIDFFeatureBuilder:
    """
    Wraps sklearn TfidfVectorizer with fit/transform helpers and
    persistence (save / load).

    Parameters
    ----------
    max_features : int
    ngram_range  : tuple
    min_df       : int
    """

    def __init__(
        self,
        max_features: int = MAX_FEATURES,
        ngram_range: tuple = NGRAM_RANGE,
        min_df: int = MIN_DF,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            strip_accents="unicode",
            analyzer="word",
            sublinear_tf=True,   # dampens high-frequency terms
        )
        self._fitted = False

    # ── Fit / transform ───────────────────────────────────────────
    def fit(self, texts: pd.Series) -> "TFIDFFeatureBuilder":
        """Fit the vectorizer on training corpus. Must be called before transform."""
        logger.info(f"Fitting TF-IDF on {len(texts):,} documents …")
        self.vectorizer.fit(texts.fillna("").astype(str))
        self._fitted = True
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        return self

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """Return a dense DataFrame of TF-IDF features."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        matrix = self.vectorizer.transform(texts.fillna("").astype(str))
        feature_names = [f"tfidf_{n}" for n in self.vectorizer.get_feature_names_out()]
        df = pd.DataFrame.sparse.from_spmatrix(matrix, columns=feature_names)
        logger.info(f"TF-IDF transform: {df.shape[0]} rows × {df.shape[1]} features")
        return df

    def fit_transform(self, texts: pd.Series) -> pd.DataFrame:
        return self.fit(texts).transform(texts)

    # ── Persistence ───────────────────────────────────────────────
    def save(self, path: str = VECTORIZER_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.vectorizer, fh)
        logger.info(f"TF-IDF vectorizer saved → {path}")

    @classmethod
    def load(cls, path: str = VECTORIZER_PATH) -> "TFIDFFeatureBuilder":
        with open(path, "rb") as fh:
            vectorizer = pickle.load(fh)
        instance = cls.__new__(cls)
        instance.vectorizer = vectorizer
        instance._fitted = True
        logger.info(f"TF-IDF vectorizer loaded ← {path}")
        return instance


# ── Convenience wrapper ───────────────────────────────────────────
def build_tfidf_features(
    train_texts: pd.Series,
    test_texts: pd.Series,
    save_path: Optional[str] = VECTORIZER_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit on train, transform train + test. Optionally persist the vectorizer.

    Returns
    -------
    train_features, test_features : pd.DataFrame (sparse)
    """
    builder = TFIDFFeatureBuilder()
    train_feats = builder.fit_transform(train_texts)
    test_feats  = builder.transform(test_texts)

    if save_path:
        builder.save(save_path)

    return train_feats, test_feats
