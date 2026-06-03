"""Multi-target model for auxiliary drilling targets.

Predicts: drilling_efficiency_score, bit_damage_risk, vibration_risk,
stuck_pipe_risk — using one LightGBM regressor per target.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from backend.utils.config import settings, CATEGORICAL_FEATURES, AUX_TARGETS
from backend.utils.features import get_model_features
from backend.utils.logger import get_logger
from backend.training.metrics import regression_metrics

log = get_logger("risk_model")

ARTIFACT = settings.model_dir / "risk_efficiency.pkl"


class RiskEfficiencyModel:
    name = "risk_efficiency"

    def __init__(self):
        self.features = get_model_features(include_categoricals=True)
        self.cat_features = CATEGORICAL_FEATURES
        self.targets = AUX_TARGETS
        self.models: dict[str, lgb.LGBMRegressor] = {}

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c in self.cat_features:
            X[c] = X[c].astype("category")
        return X

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        X = self._prep(train)
        params = dict(objective="regression", metric="rmse", n_estimators=400,
                      learning_rate=0.05, num_leaves=48, subsample=0.85,
                      colsample_bytree=0.85, random_state=settings.random_seed,
                      n_jobs=-1, verbose=-1)
        for t in self.targets:
            m = lgb.LGBMRegressor(**params)
            es = None
            if valid is not None:
                es = [(self._prep(valid), valid[t])]
            m.fit(X, train[t], eval_set=es, categorical_feature=self.cat_features,
                  callbacks=[lgb.early_stopping(40, verbose=False)] if es else None)
            self.models[t] = m
            log.info("Trained aux target: %s", t)
        return self

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = self._prep(df)
        out = {t: np.clip(m.predict(X), 0, 1) for t, m in self.models.items()}
        # efficiency may exceed 1 slightly; clip already handles 0..1
        return out

    def evaluate(self, df: pd.DataFrame) -> dict[str, dict]:
        preds = self.predict(df)
        return {t: regression_metrics(df[t], preds[t]) for t in self.targets}

    def save(self, path: Path | None = None):
        path = path or ARTIFACT
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "features": self.features,
                     "cat_features": self.cat_features, "targets": self.targets}, path)
        log.info("Saved RiskEfficiency -> %s", path)

    @classmethod
    def load(cls, path: Path | None = None) -> "RiskEfficiencyModel":
        path = path or ARTIFACT
        blob = joblib.load(path)
        obj = cls()
        obj.models = blob["models"]
        obj.features = blob["features"]
        obj.cat_features = blob["cat_features"]
        obj.targets = blob["targets"]
        return obj
