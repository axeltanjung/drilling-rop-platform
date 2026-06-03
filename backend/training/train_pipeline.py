"""End-to-end ML training pipeline.

Stages:
  1. Load + preprocess + feature engineering
  2. Time-aware train/test split
  3. Train LightGBM (with optional Optuna tuning) — primary ROP model
  4. Train CatBoost — benchmark
  5. Train Deep model (MLP) — non-linear
  6. Train Risk/Efficiency multi-target model
  7. Evaluate all (RMSE / MAE / R2) and log to MLflow
  8. Persist artifacts + leaderboard JSON

Run:
    python backend/training/train_pipeline.py            # full
    python backend/training/train_pipeline.py --fast     # smaller/faster
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.utils.config import settings, TARGET  # noqa: E402
from backend.utils.logger import get_logger  # noqa: E402
from backend.training.preprocess import preprocess, train_test_split_by_well  # noqa: E402
from backend.training import mlflow_tracking as mlf  # noqa: E402
from backend.models.lightgbm_model import LightGBMROPModel  # noqa: E402
from backend.models.catboost_model import CatBoostROPModel  # noqa: E402
from backend.models.deep_model import DeepROPModel  # noqa: E402
from backend.models.risk_model import RiskEfficiencyModel  # noqa: E402

log = get_logger("train_pipeline")


def run(fast: bool = False, tune: bool = True):
    mlf.init_tracking()
    df = preprocess(save=True)
    if fast:
        df = df.groupby("well_id").head(4000).reset_index(drop=True)
        tune = False
    train, test = train_test_split_by_well(df, test_frac=0.2)
    leaderboard = {}

    # ---------- LightGBM (primary) ----------
    log.info("=== Training LightGBM ===")
    with mlf.start_run("lightgbm_rop"):
        lgbm = LightGBMROPModel()
        if tune:
            lgbm.tune(train, test, TARGET, n_trials=20)
        lgbm.fit(train, TARGET, valid=test)
        m = lgbm.evaluate(test, TARGET)
        mlf.log_params(lgbm.params)
        mlf.log_metrics(m, prefix="test_")
        lgbm.save()
        leaderboard["lightgbm"] = m
        log.info("LightGBM test metrics: %s", m)

    # ---------- CatBoost (benchmark) ----------
    log.info("=== Training CatBoost ===")
    with mlf.start_run("catboost_rop"):
        cat = CatBoostROPModel()
        cat.fit(train, TARGET, valid=test)
        m = cat.evaluate(test, TARGET)
        mlf.log_params(cat.params)
        mlf.log_metrics(m, prefix="test_")
        cat.save()
        leaderboard["catboost"] = m
        log.info("CatBoost test metrics: %s", m)

    # ---------- Deep (MLP) ----------
    log.info("=== Training Deep MLP ===")
    try:
        with mlf.start_run("deep_mlp_rop"):
            deep = DeepROPModel(arch="mlp")
            deep.fit(train, TARGET, valid=test, epochs=8 if fast else 18)
            m = deep.evaluate(test, TARGET)
            mlf.log_metrics(m, prefix="test_")
            deep.save()
            leaderboard["deep_mlp"] = m
            log.info("Deep MLP test metrics: %s", m)
    except Exception as e:  # TF may be unavailable in some envs
        log.warning("Deep model training skipped: %s", e)

    # ---------- Risk / Efficiency ----------
    log.info("=== Training Risk/Efficiency multi-target ===")
    with mlf.start_run("risk_efficiency"):
        risk = RiskEfficiencyModel()
        risk.fit(train, valid=test)
        rm = risk.evaluate(test)
        for tgt, metrics in rm.items():
            mlf.log_metrics(metrics, prefix=f"{tgt}_")
        risk.save()
        leaderboard["risk_efficiency"] = rm
        log.info("Risk/Efficiency metrics: %s", rm)

    # ---------- leaderboard ----------
    lb_path = settings.model_dir / "leaderboard.json"
    lb_path.write_text(json.dumps(leaderboard, indent=2))
    log.info("Leaderboard saved -> %s", lb_path)

    # pick best ROP model by RMSE
    rop_models = {k: v for k, v in leaderboard.items() if k in ("lightgbm", "catboost", "deep_mlp")}
    best = min(rop_models, key=lambda k: rop_models[k]["rmse"])
    (settings.model_dir / "best_model.txt").write_text(best)
    log.info("BEST ROP MODEL: %s (rmse=%.3f)", best, rop_models[best]["rmse"])
    return leaderboard


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller subset, no tuning")
    ap.add_argument("--no-tune", action="store_true", help="skip Optuna tuning")
    args = ap.parse_args()
    run(fast=args.fast, tune=not args.no_tune)
