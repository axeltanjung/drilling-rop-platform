"""MLflow tracking helper — thin wrapper with safe no-op fallback."""
from __future__ import annotations

from contextlib import contextmanager

from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger("mlflow_tracking")

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:  # pragma: no cover
    _HAS_MLFLOW = False
    log.warning("mlflow not installed; tracking disabled")


def init_tracking():
    if not _HAS_MLFLOW:
        return
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    log.info("MLflow tracking URI=%s experiment=%s",
             settings.mlflow_tracking_uri, settings.mlflow_experiment)


@contextmanager
def start_run(run_name: str):
    if not _HAS_MLFLOW:
        yield None
        return
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_params(params: dict):
    if _HAS_MLFLOW and params:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})


def log_metrics(metrics: dict, prefix: str = ""):
    if _HAS_MLFLOW:
        mlflow.log_metrics({f"{prefix}{k}": float(v) for k, v in metrics.items()})


def log_artifact(path: str):
    if _HAS_MLFLOW:
        mlflow.log_artifact(path)


def register_model_sklearn(model, name: str):
    """Register a sklearn-flavored model into the MLflow registry."""
    if not _HAS_MLFLOW:
        return
    try:
        mlflow.sklearn.log_model(model, artifact_path=name, registered_model_name=name)
        log.info("Registered model in MLflow registry: %s", name)
    except Exception as e:  # pragma: no cover
        log.warning("Model registry logging failed for %s: %s", name, e)
