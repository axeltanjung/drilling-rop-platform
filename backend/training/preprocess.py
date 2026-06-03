"""Data preprocessing for model training and inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.utils.config import settings, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from backend.utils.features import add_engineered_features
from backend.utils.logger import get_logger

log = get_logger("preprocess")


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or settings.raw_data_path
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run synthetic_drilling_data_generator.py first."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    log.info("Loaded raw data: %s rows", f"{len(df):,}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and clip gross outliers."""
    df = df.copy()

    # Median imputation for numeric sensors with missing values
    for col in NUMERIC_FEATURES:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)

    # Clip numeric features to robust percentile bounds to tame outliers
    for col in NUMERIC_FEATURES:
        lo, hi = df[col].quantile([0.001, 0.999])
        df[col] = df[col].clip(lo, hi)

    # Fill any categorical gaps
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode().iloc[0]).astype(str)

    return df


def preprocess(df: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    """Full preprocessing: clean + feature engineering."""
    if df is None:
        df = load_raw()
    df = clean(df)
    df = add_engineered_features(df)
    log.info("Preprocessed data: %d rows x %d cols", df.shape[0], df.shape[1])

    if save:
        out = settings.processed_dir / "drilling_processed.csv"
        df.to_csv(out, index=False)
        log.info("Saved processed -> %s", out)
    return df


def train_test_split_by_well(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Time-aware split: hold out the latest portion of each well to avoid leakage."""
    train_parts, test_parts = [], []
    for _, g in df.sort_values("timestamp").groupby("well_id"):
        cut = int(len(g) * (1 - test_frac))
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    log.info("Split -> train=%d test=%d", len(train), len(test))
    return train, test


if __name__ == "__main__":
    preprocess()
