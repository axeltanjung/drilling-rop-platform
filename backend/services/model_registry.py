"""Model registry / lazy loader.

Loads trained model artifacts once and caches them. Falls back gracefully
if artifacts are missing (returns informative errors to the API layer).
"""
from __future__ import annotations

from functools import lru_cache

from backend.utils.config import settings
from backend.utils.logger import get_logger
from backend.models.lightgbm_model import LightGBMROPModel
from backend.models.risk_model import RiskEfficiencyModel

log = get_logger("model_registry")


class ModelNotTrainedError(RuntimeError):
    pass


def best_model_name() -> str:
    f = settings.model_dir / "best_model.txt"
    return f.read_text().strip() if f.exists() else "lightgbm"


@lru_cache(maxsize=1)
def get_rop_model():
    name = best_model_name()
    try:
        if name == "catboost":
            from backend.models.catboost_model import CatBoostROPModel
            return CatBoostROPModel.load()
        if name == "deep_mlp":
            from backend.models.deep_model import DeepROPModel
            return DeepROPModel.load()
        return LightGBMROPModel.load()
    except FileNotFoundError as e:
        raise ModelNotTrainedError(
            "ROP model artifact not found. Run training/train_pipeline.py first."
        ) from e


@lru_cache(maxsize=1)
def get_lightgbm_model() -> LightGBMROPModel:
    """Always-tree model used for SHAP explainability."""
    try:
        return LightGBMROPModel.load()
    except FileNotFoundError as e:
        raise ModelNotTrainedError("LightGBM artifact not found. Train models first.") from e


@lru_cache(maxsize=1)
def get_risk_model() -> RiskEfficiencyModel:
    try:
        return RiskEfficiencyModel.load()
    except FileNotFoundError as e:
        raise ModelNotTrainedError("Risk model artifact not found. Train models first.") from e


@lru_cache(maxsize=1)
def get_optimizer():
    from backend.optimization.optimizer import DrillingOptimizer
    return DrillingOptimizer(get_rop_model(), get_risk_model())


@lru_cache(maxsize=1)
def get_explainer():
    from backend.explainability.explainer import ROPExplainer
    # explainer uses tree model for SHAP
    return ROPExplainer(get_lightgbm_model())


def models_available() -> bool:
    return (settings.model_dir / "lightgbm_rop.pkl").exists()
