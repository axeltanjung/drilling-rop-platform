"""Smoke + unit tests for the drilling platform.

These are designed to run even before models are trained:
  * data generator produces a valid small dataset
  * feature engineering is null-safe
  * preprocessing cleans data
  * API health endpoint works
Model-dependent tests are skipped if artifacts are missing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.data.synthetic_drilling_data_generator import generate
from backend.utils.features import add_engineered_features, ENGINEERED_FEATURES
from backend.training.preprocess import clean
from backend.services import model_registry as reg


@pytest.fixture(scope="module")
def small_df():
    return generate(n_rows=2000, n_wells=3, save=False)


def test_generator_shape_and_targets(small_df):
    assert len(small_df) == 2000
    assert small_df["well_id"].nunique() == 3
    assert small_df["rate_of_penetration"].between(0, 200).all()
    for col in ["bit_damage_risk", "vibration_risk", "stuck_pipe_risk",
                "drilling_efficiency_score"]:
        assert small_df[col].between(0, 1).all()


def test_feature_engineering_no_nan(small_df):
    out = add_engineered_features(small_df)
    for f in ENGINEERED_FEATURES:
        assert f in out.columns
        assert not out[f].isna().any()
        assert np.isfinite(out[f]).all()


def test_clean_imputes_missing():
    df = generate(n_rows=1000, n_wells=2, save=False)
    df.loc[df.index[:50], "mud_density"] = np.nan
    cleaned = clean(df)
    assert not cleaned["mud_density"].isna().any()


def test_physics_hardness_reduces_rop():
    """Granite (hard) should drill slower than Sandstone (soft) on average."""
    df = generate(n_rows=20000, n_wells=6, save=False)
    soft = df[df.formation_type == "Sandstone"]["rate_of_penetration"].mean()
    hard = df[df.formation_type == "Granite"]["rate_of_penetration"].mean()
    assert soft > hard


def test_api_health():
    from fastapi.testclient import TestClient
    from backend.api.main import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.skipif(not reg.models_available(), reason="models not trained")
def test_predict_rop_endpoint():
    from fastapi.testclient import TestClient
    from backend.api.main import app
    client = TestClient(app)
    payload = {
        "depth": 8500, "formation_type": "Shale", "bit_type": "PDC",
        "weight_on_bit": 28, "rpm": 120, "torque": 35, "mud_flow_rate": 600,
    }
    r = client.post("/predict/rop", json=payload)
    assert r.status_code == 200
    assert r.json()["predicted_rop"] > 0
