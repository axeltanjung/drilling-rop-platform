"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from backend.utils.config import FORMATION_TYPES, BIT_TYPES


class DrillingConditions(BaseModel):
    """Full set of drilling features required for prediction/optimization."""
    depth: float = Field(..., ge=0, le=40000, examples=[8500])
    formation_type: str = Field(..., examples=["Shale"])
    bit_type: str = Field(..., examples=["PDC"])
    weight_on_bit: float = Field(..., ge=0, le=80, examples=[28])
    rpm: float = Field(..., ge=0, le=300, examples=[120])
    torque: float = Field(..., ge=0, le=200, examples=[35])
    mud_flow_rate: float = Field(..., ge=0, le=1200, examples=[600])
    standpipe_pressure: float = Field(2800, ge=0, le=8000)
    hook_load: float = Field(220, ge=0, le=800)
    bit_wear: float = Field(0.2, ge=0, le=1)
    mud_density: float = Field(11.5, ge=5, le=20)
    vibration_level: float = Field(2.0, ge=0, le=10)
    temperature: float = Field(160, ge=50, le=400)
    drilling_hours: float = Field(40, ge=0)
    pump_pressure: float = Field(2700, ge=0, le=8000)
    flow_out: float = Field(580, ge=0, le=1200)
    differential_pressure: float = Field(80, ge=-500, le=1000)

    def to_features(self) -> dict:
        return self.model_dump()


class ROPResponse(BaseModel):
    predicted_rop: float
    unit: str = "ft/hr"
    model: str


class EfficiencyResponse(BaseModel):
    drilling_efficiency_score: float
    rating: str


class RiskResponse(BaseModel):
    vibration_risk: float
    bit_damage_risk: float
    stuck_pipe_risk: float
    instability_index: float
    risk_level: str


class OptimizeResponse(BaseModel):
    baseline_rop: float
    optimized_rop: float
    improvement_pct: float
    recommended_parameters: dict
    baseline_parameters: dict
    risk_before: dict
    risk_after: dict
    trade_offs: list[str]


class ExplainResponse(BaseModel):
    predicted_rop: float
    base_value: float
    confidence: Optional[dict] = None
    contributions: list[dict]
    narrative: str


class HealthResponse(BaseModel):
    status: str
    models_available: bool
    version: str
