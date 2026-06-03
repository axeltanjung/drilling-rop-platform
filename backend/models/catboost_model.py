"""CatBoost ROP model wrapper.

Advanced gradient boosting benchmark with native categorical handling
and feature-interaction analysis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from backend.utils.config import settings, CATEGORICAL_FEATURES
from backend.utils.features import get_model_features
from backend.utils.logger import get_logger
from backend.training.metrics import regression_metrics

log = get_logger("catboost_model")

ARTIFACT = settings.model_dir / "catboost_rop.cbm"


class CatBoostROPModel:
    name = "catboost"

    def __init__(self, params: dict | None = None):
        self.features = get_model_features(include_categoricals=True)
        self.cat_features = CATEGORICAL_FEATURES
        self.params = params or {
            "iterations": 700,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
            "random_seed": settings.random_seed,
            "verbose": False,
        }
        self.model: CatBoostRegressor | None = None

    def _pool(self, df: pd.DataFrame, target: str | None = None) -> Pool:
        X = df[self.features].copy()
        for c in self.cat_features:
            X[c] = X[c].astype(str)
        label = df[target] if target else None
        return Pool(X, label=label, cat_features=self.cat_features)

    def fit(self, train: pd.DataFrame, target: str, valid: pd.DataFrame | None = None):
        self.model = CatBoostRegressor(**self.params)
        eval_pool = self._pool(valid, target) if valid is not None else None
        self.model.fit(self._pool(train, target), eval_set=eval_pool,
                       use_best_model=eval_pool is not None)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(self._pool(df))

    def evaluate(self, df: pd.DataFrame, target: str) -> dict:
        return regression_metrics(df[target], self.predict(df))

    def feature_importance(self) -> dict[str, float]:
        assert self.model is not None
        imp = self.model.get_feature_importance()
        return dict(sorted(zip(self.features, imp.tolist()), key=lambda x: -x[1]))

    def interaction_strength(self, df: pd.DataFrame, target: str, top: int = 8):
        """Return top pairwise feature interactions."""
        assert self.model is not None
        try:
            interactions = self.model.get_feature_importance(
                data=self._pool(df, target), type="Interaction"
            )
            named = [
                {
                    "feature_a": self.features[int(a)],
                    "feature_b": self.features[int(b)],
                    "strength": float(score),
                }
                for a, b, score in interactions[:top]
            ]
            return named
        except Exception as e:  # pragma: no cover
            log.warning("Interaction analysis failed: %s", e)
            return []

    def save(self, path: Path | None = None):
        path = path or ARTIFACT
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        # persist feature metadata alongside
        meta = path.with_suffix(".meta.npy")
        np.save(meta, {"features": self.features, "cat_features": self.cat_features,
                       "params": self.params}, allow_pickle=True)
        log.info("Saved CatBoost -> %s", path)

    @classmethod
    def load(cls, path: Path | None = None) -> "CatBoostROPModel":
        path = path or ARTIFACT
        meta = np.load(path.with_suffix(".meta.npy"), allow_pickle=True).item()
        obj = cls(params=meta["params"])
        obj.features = meta["features"]
        obj.cat_features = meta["cat_features"]
        obj.model = CatBoostRegressor()
        obj.model.load_model(str(path))
        return obj
