"""Feature engineering shared between training and inference.

Adds physics-inspired interaction features and deterministically
encodes categorical variables so train/serve stay consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.utils.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Engineered feature names produced by `add_engineered_features`
ENGINEERED_FEATURES = [
    "mechanical_specific_energy",
    "wob_rpm_product",
    "torque_per_wob",
    "hydraulic_power",
    "flow_balance",
    "bit_aggressiveness",
    "vibration_load_index",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add drilling-physics-inspired interaction features.

    All operations are vectorized and null-safe so this works for both
    bulk training and single-row inference payloads.
    """
    out = df.copy()
    eps = 1e-6

    wob = out["weight_on_bit"].astype(float)
    rpm = out["rpm"].astype(float)
    torque = out["torque"].astype(float)
    flow_in = out["mud_flow_rate"].astype(float)
    flow_out = out["flow_out"].astype(float)
    pump_p = out["pump_pressure"].astype(float)
    rop = out.get("rate_of_penetration", pd.Series(1.0, index=out.index)).astype(float)

    # Mechanical Specific Energy (proxy) — energy to remove unit rock volume
    out["mechanical_specific_energy"] = (
        wob / (np.pi + eps) + (120.0 * np.pi * rpm * torque) / ((rop + eps) * 8.0)
    )
    out["wob_rpm_product"] = wob * rpm
    out["torque_per_wob"] = torque / (wob + eps)
    out["hydraulic_power"] = pump_p * flow_in / 1714.0
    out["flow_balance"] = flow_in - flow_out
    out["bit_aggressiveness"] = (wob * rpm) / (torque + eps)
    out["vibration_load_index"] = out["vibration_level"].astype(float) * (wob + eps) / (rpm + eps)

    # Clean any inf/nan from divisions
    out[ENGINEERED_FEATURES] = (
        out[ENGINEERED_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    return out


def get_model_features(include_categoricals: bool = True) -> list[str]:
    feats = NUMERIC_FEATURES + ENGINEERED_FEATURES
    if include_categoricals:
        feats = feats + CATEGORICAL_FEATURES
    return feats


def encode_categoricals(df: pd.DataFrame, mappings: dict[str, dict]) -> pd.DataFrame:
    """Integer-encode categorical columns using provided mappings."""
    out = df.copy()
    for col, mapping in mappings.items():
        if col in out.columns:
            out[col] = out[col].map(mapping).fillna(-1).astype(int)
    return out


def build_category_mappings(df: pd.DataFrame) -> dict[str, dict]:
    mappings: dict[str, dict] = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            cats = sorted(df[col].astype(str).unique())
            mappings[col] = {c: i for i, c in enumerate(cats)}
    return mappings
