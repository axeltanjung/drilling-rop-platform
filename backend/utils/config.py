"""Centralized configuration loaded from environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

# Project root = two levels up from this file (backend/utils/config.py -> root)
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """Application settings sourced from environment with sensible defaults."""

    def __init__(self) -> None:
        self.root_dir: Path = ROOT_DIR

        # API
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # Database
        self.database_url: str = os.getenv(
            "DATABASE_URL", f"sqlite:///{ROOT_DIR / 'data' / 'drilling.db'}"
        )

        # Paths
        self.data_dir: Path = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))
        self.raw_data_path: Path = Path(
            os.getenv("RAW_DATA_PATH", str(self.data_dir / "raw" / "drilling_telemetry.csv"))
        )
        self.processed_dir: Path = self.data_dir / "processed"
        self.model_dir: Path = Path(os.getenv("MODEL_DIR", str(ROOT_DIR / "backend" / "models" / "artifacts")))

        # MLflow
        self.mlflow_tracking_uri: str = os.getenv(
            "MLFLOW_TRACKING_URI", str(ROOT_DIR / "mlflow" / "mlruns")
        )
        self.mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT", "drilling_rop")

        # Data generation
        self.n_wells: int = int(os.getenv("N_WELLS", "12"))
        self.n_rows: int = int(os.getenv("N_ROWS", "200000"))
        self.random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

        # Ensure directories exist
        for d in [self.data_dir, self.data_dir / "raw", self.processed_dir, self.model_dir]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Canonical feature lists shared across training & inference
NUMERIC_FEATURES = [
    "depth",
    "weight_on_bit",
    "rpm",
    "torque",
    "mud_flow_rate",
    "standpipe_pressure",
    "hook_load",
    "bit_wear",
    "mud_density",
    "vibration_level",
    "temperature",
    "drilling_hours",
    "pump_pressure",
    "flow_out",
    "differential_pressure",
]

CATEGORICAL_FEATURES = ["formation_type", "bit_type"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "rate_of_penetration"

AUX_TARGETS = [
    "drilling_efficiency_score",
    "bit_damage_risk",
    "vibration_risk",
    "stuck_pipe_risk",
]

FORMATION_TYPES = ["Sandstone", "Shale", "Limestone", "Dolomite", "Granite", "Salt"]
BIT_TYPES = ["PDC", "Tricone", "Diamond", "Hybrid"]
