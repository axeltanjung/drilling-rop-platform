"""Dashboard & well analytics service — reads predictions from SQLite."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.services.database import query_df, table_exists
from backend.utils.logger import get_logger

log = get_logger("dashboard_service")

# Rough cost model for indicators (USD)
RIG_DAY_RATE = 250_000.0  # USD/day


def _no_data() -> dict:
    return {"available": False, "message": "No predictions found. Run training/batch_predict.py."}


def executive_summary() -> dict:
    if not table_exists("predictions"):
        return _no_data()
    df = query_df("SELECT * FROM predictions")
    if df.empty:
        return _no_data()

    avg_rop = float(df["pred_rop"].mean())
    avg_eff = float(df["pred_drilling_efficiency_score"].mean())
    active_wells = int(df["well_id"].nunique())
    op_risk = float(np.mean([
        df["pred_vibration_risk"].mean(),
        df["pred_bit_damage_risk"].mean(),
        df["pred_stuck_pipe_risk"].mean(),
    ]))

    # crude cost indicator: faster ROP -> fewer days -> lower cost index (0-100, lower better)
    cost_index = float(np.clip(100 - (avg_rop / 80.0) * 100, 0, 100))

    # formation comparison
    form = (df.groupby("formation_type")
              .agg(avg_rop=("pred_rop", "mean"),
                   avg_efficiency=("pred_drilling_efficiency_score", "mean"),
                   count=("formation_type", "size"))
              .reset_index().round(2))

    # ROP trend by depth bucket
    df["depth_bucket"] = (df["depth"] // 500 * 500).astype(int)
    trend = (df.groupby("depth_bucket")
               .agg(avg_rop=("pred_rop", "mean"))
               .reset_index().sort_values("depth_bucket").round(2))

    return {
        "available": True,
        "kpis": {
            "avg_rop": round(avg_rop, 2),
            "avg_efficiency": round(avg_eff * 100, 1),
            "active_wells": active_wells,
            "operational_risk": round(op_risk * 100, 1),
            "cost_index": round(cost_index, 1),
            "total_records": int(len(df)),
        },
        "formation_comparison": form.to_dict(orient="records"),
        "rop_trend": trend.to_dict(orient="records"),
        "wells": well_list(),
    }


def well_list() -> list[dict]:
    if not table_exists("well_summary"):
        return []
    return query_df("SELECT * FROM well_summary ORDER BY well_id").round(3).to_dict(orient="records")


def well_detail(well_id: str, max_points: int = 1000) -> dict:
    if not table_exists("predictions"):
        return _no_data()
    df = query_df("SELECT * FROM predictions WHERE well_id = :w ORDER BY depth", {"w": well_id})
    if df.empty:
        return {"available": False, "message": f"Well {well_id} not found"}

    # downsample for charting
    if len(df) > max_points:
        idx = np.linspace(0, len(df) - 1, max_points).astype(int)
        df = df.iloc[idx]

    series = df[[
        "timestamp", "depth", "pred_rop", "rate_of_penetration", "weight_on_bit",
        "rpm", "torque", "bit_wear", "vibration_level",
        "pred_vibration_risk", "pred_bit_damage_risk", "pred_stuck_pipe_risk",
    ]].copy()
    series["timestamp"] = series["timestamp"].astype(str)

    formations = (query_df(
        "SELECT formation_type, MIN(depth) lo, MAX(depth) hi FROM predictions "
        "WHERE well_id = :w GROUP BY formation_type", {"w": well_id})
        .round(1).to_dict(orient="records"))

    return {
        "available": True,
        "well_id": well_id,
        "summary": {
            "avg_rop": round(float(df["pred_rop"].mean()), 2),
            "max_depth": round(float(df["depth"].max()), 1),
            "avg_bit_wear": round(float(df["bit_wear"].mean()), 3),
            "points": int(len(series)),
        },
        "series": series.round(3).to_dict(orient="records"),
        "formations": formations,
    }


def risk_overview() -> dict:
    if not table_exists("predictions"):
        return _no_data()
    df = query_df(
        "SELECT well_id, formation_type, "
        "AVG(pred_stuck_pipe_risk) stuck, AVG(pred_vibration_risk) vibration, "
        "AVG(pred_bit_damage_risk) bit_damage "
        "FROM predictions GROUP BY well_id, formation_type")
    by_well = (df.groupby("well_id")[["stuck", "vibration", "bit_damage"]]
                 .mean().reset_index().round(3))
    by_well["instability_index"] = (
        by_well[["stuck", "vibration", "bit_damage"]].mean(axis=1).round(3))
    return {
        "available": True,
        "by_well": by_well.sort_values("instability_index", ascending=False)
                          .to_dict(orient="records"),
        "by_formation": (df.groupby("formation_type")[["stuck", "vibration", "bit_damage"]]
                           .mean().reset_index().round(3).to_dict(orient="records")),
    }
