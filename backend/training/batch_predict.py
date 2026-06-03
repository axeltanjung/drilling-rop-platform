"""Batch prediction pipeline.

Scores the processed dataset (or an uploaded drilling report) with the
trained models and persists results into SQLite for the API/dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.utils.config import settings, TARGET  # noqa: E402
from backend.utils.logger import get_logger  # noqa: E402
from backend.utils.features import add_engineered_features  # noqa: E402
from backend.models.lightgbm_model import LightGBMROPModel  # noqa: E402
from backend.models.risk_model import RiskEfficiencyModel  # noqa: E402
from backend.services.database import init_db, get_engine  # noqa: E402

log = get_logger("batch_predict")


def _load_best_rop_model():
    best_file = settings.model_dir / "best_model.txt"
    best = best_file.read_text().strip() if best_file.exists() else "lightgbm"
    if best == "catboost":
        from backend.models.catboost_model import CatBoostROPModel
        return CatBoostROPModel.load()
    if best == "deep_mlp":
        try:
            from backend.models.deep_model import DeepROPModel
            return DeepROPModel.load()
        except Exception:
            return LightGBMROPModel.load()
    return LightGBMROPModel.load()


def run(df: pd.DataFrame | None = None, persist: bool = True) -> pd.DataFrame:
    if df is None:
        proc = settings.processed_dir / "drilling_processed.csv"
        if proc.exists():
            df = pd.read_csv(proc, parse_dates=["timestamp"])
        else:
            df = pd.read_csv(settings.raw_data_path, parse_dates=["timestamp"])
            df = add_engineered_features(df)
    else:
        df = add_engineered_features(df)

    rop_model = _load_best_rop_model()
    risk_model = RiskEfficiencyModel.load()

    log.info("Scoring %d rows...", len(df))
    df = df.copy()
    df["pred_rop"] = rop_model.predict(df)
    risk_preds = risk_model.predict(df)
    for t, vals in risk_preds.items():
        df[f"pred_{t}"] = vals

    if persist:
        init_db()
        engine = get_engine()
        keep = [
            "timestamp", "well_id", "depth", "formation_type", "bit_type",
            "weight_on_bit", "rpm", "torque", "mud_flow_rate", "bit_wear",
            "vibration_level", "rate_of_penetration", "pred_rop",
            "pred_drilling_efficiency_score", "pred_bit_damage_risk",
            "pred_vibration_risk", "pred_stuck_pipe_risk",
        ]
        out = df[[c for c in keep if c in df.columns]].copy()
        out.to_sql("predictions", engine, if_exists="replace", index=False)

        # Well-level summary table
        summary = (
            out.groupby("well_id")
            .agg(
                rows=("well_id", "size"),
                avg_rop=("pred_rop", "mean"),
                max_depth=("depth", "max"),
                avg_efficiency=("pred_drilling_efficiency_score", "mean"),
                avg_vibration_risk=("pred_vibration_risk", "mean"),
                avg_bit_damage_risk=("pred_bit_damage_risk", "mean"),
                avg_stuck_pipe_risk=("pred_stuck_pipe_risk", "mean"),
            )
            .reset_index()
        )
        summary.to_sql("well_summary", engine, if_exists="replace", index=False)
        log.info("Persisted predictions (%d) + well_summary (%d) to SQLite",
                 len(out), len(summary))

    return df


if __name__ == "__main__":
    run()
