"""LightGBM ROP prediction model wrapper.

Primary tabular model with hyperparameter tuning (Optuna) and SHAP support.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb

from backend.utils.config import settings, CATEGORICAL_FEATURES
from backend.utils.features import get_model_features
from backend.utils.logger import get_logger
from backend.training.metrics import regression_metrics

log = get_logger("lightgbm_model")

ARTIFACT = settings.model_dir / "lightgbm_rop.pkl"


class LightGBMROPModel:
    name = "lightgbm"

    def __init__(self, params: dict | None = None):
        self.features = get_model_features(include_categoricals=True)
        self.cat_features = CATEGORICAL_FEATURES
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 600,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": -1,
            "min_child_samples": 40,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 0.2,
            "random_state": settings.random_seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.model: lgb.LGBMRegressor | None = None

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c in self.cat_features:
            X[c] = X[c].astype("category")
        return X

    def fit(self, train: pd.DataFrame, target: str, valid: pd.DataFrame | None = None):
        X = self._prep(train)
        y = train[target]
        eval_set = None
        if valid is not None:
            eval_set = [(self._prep(valid), valid[target])]
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X, y,
            eval_set=eval_set,
            categorical_feature=self.cat_features,
            callbacks=[lgb.early_stopping(50, verbose=False)] if eval_set else None,
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model not trained/loaded"
        return self.model.predict(self._prep(df))

    def evaluate(self, df: pd.DataFrame, target: str) -> dict:
        return regression_metrics(df[target], self.predict(df))

    def feature_importance(self) -> dict[str, float]:
        assert self.model is not None
        imp = self.model.feature_importances_
        return dict(sorted(zip(self.features, imp.tolist()), key=lambda x: -x[1]))

    def tune(self, train: pd.DataFrame, valid: pd.DataFrame, target: str, n_trials: int = 25):
        """Optuna hyperparameter tuning (optional, called by pipeline)."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        Xtr, ytr = self._prep(train), train[target]
        Xva, yva = self._prep(valid), valid[target]

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": 800,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "random_state": settings.random_seed,
                "n_jobs": -1,
                "verbose": -1,
            }
            m = lgb.LGBMRegressor(**params)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  categorical_feature=self.cat_features,
                  callbacks=[lgb.early_stopping(40, verbose=False)])
            return regression_metrics(yva, m.predict(Xva))["rmse"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        log.info("Best LGBM params: %s (rmse=%.3f)", study.best_params, study.best_value)
        self.params.update(study.best_params)
        return study.best_params

    def save(self, path: Path | None = None):
        path = path or ARTIFACT
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features,
                     "cat_features": self.cat_features, "params": self.params}, path)
        log.info("Saved LightGBM -> %s", path)

    @classmethod
    def load(cls, path: Path | None = None) -> "LightGBMROPModel":
        path = path or ARTIFACT
        blob = joblib.load(path)
        obj = cls(params=blob["params"])
        obj.model = blob["model"]
        obj.features = blob["features"]
        obj.cat_features = blob["cat_features"]
        return obj
