"""Drilling Parameter Optimization Engine.

Given current drilling conditions, search the controllable parameter space
(Weight-on-Bit, RPM, Mud Flow Rate) to recommend settings that maximize
predicted ROP while penalizing predicted vibration and bit-damage risk.

Approach: surrogate-model-based grid/random search over the trained ROP and
Risk/Efficiency models. Lightweight, deterministic and fully local.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backend.utils.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from backend.utils.features import add_engineered_features
from backend.utils.logger import get_logger

log = get_logger("optimizer")

# Controllable parameter bounds (operational envelope)
PARAM_BOUNDS = {
    "weight_on_bit": (8.0, 50.0),
    "rpm": (60.0, 200.0),
    "mud_flow_rate": (350.0, 850.0),
}


@dataclass
class OptimizationResult:
    baseline_rop: float
    optimized_rop: float
    improvement_pct: float
    recommended: dict
    baseline_params: dict
    risk_before: dict
    risk_after: dict
    trade_offs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "baseline_rop": round(self.baseline_rop, 2),
            "optimized_rop": round(self.optimized_rop, 2),
            "improvement_pct": round(self.improvement_pct, 2),
            "recommended_parameters": {k: round(v, 1) for k, v in self.recommended.items()},
            "baseline_parameters": {k: round(v, 1) for k, v in self.baseline_params.items()},
            "risk_before": {k: round(v, 3) for k, v in self.risk_before.items()},
            "risk_after": {k: round(v, 3) for k, v in self.risk_after.items()},
            "trade_offs": self.trade_offs,
        }


class DrillingOptimizer:
    def __init__(self, rop_model, risk_model,
                 vib_penalty: float = 18.0, bit_penalty: float = 14.0):
        self.rop_model = rop_model
        self.risk_model = risk_model
        self.vib_penalty = vib_penalty
        self.bit_penalty = bit_penalty

    def _objective(self, scored: pd.DataFrame) -> np.ndarray:
        """Penalized ROP objective (higher is better)."""
        rop = scored["pred_rop"].values
        vib = scored["pred_vibration_risk"].values
        bit = scored["pred_bit_damage_risk"].values
        return rop - self.vib_penalty * vib - self.bit_penalty * bit

    def _score_grid(self, base_row: pd.Series, grid: list[dict]) -> pd.DataFrame:
        rows = []
        for combo in grid:
            r = base_row.copy()
            for k, v in combo.items():
                r[k] = v
            rows.append(r)
        cand = pd.DataFrame(rows)
        cand = add_engineered_features(cand)
        cand["pred_rop"] = self.rop_model.predict(cand)
        risk = self.risk_model.predict(cand)
        for t, vals in risk.items():
            cand[f"pred_{t}"] = vals
        return cand

    def optimize(self, conditions: dict, resolution: int = 7) -> OptimizationResult:
        """Optimize controllable params given fixed downhole conditions.

        `conditions` must contain all NUMERIC_FEATURES + CATEGORICAL_FEATURES.
        """
        base_row = pd.Series(conditions)

        # baseline (current params)
        base_df = add_engineered_features(pd.DataFrame([base_row]))
        base_df["pred_rop"] = self.rop_model.predict(base_df)
        brisk = self.risk_model.predict(base_df)
        for t, vals in brisk.items():
            base_df[f"pred_{t}"] = vals
        baseline_rop = float(base_df["pred_rop"].iloc[0])
        risk_before = {t: float(base_df[f"pred_{t}"].iloc[0]) for t in brisk}

        # grid over controllable params
        axes = {}
        for p, (lo, hi) in PARAM_BOUNDS.items():
            axes[p] = np.linspace(lo, hi, resolution)
        grid = [dict(zip(axes.keys(), vals)) for vals in itertools.product(*axes.values())]
        log.info("Optimizing over %d parameter combinations", len(grid))

        cand = self._score_grid(base_row, grid)
        obj = self._objective(cand)
        best_idx = int(np.argmax(obj))
        best = cand.iloc[best_idx]

        recommended = {p: float(best[p]) for p in PARAM_BOUNDS}
        optimized_rop = float(best["pred_rop"])
        risk_after = {t: float(best[f"pred_{t}"]) for t in brisk}
        improvement = (optimized_rop - baseline_rop) / max(baseline_rop, 1e-6) * 100

        trade_offs = self._build_trade_offs(conditions, recommended, risk_before, risk_after)

        return OptimizationResult(
            baseline_rop=baseline_rop,
            optimized_rop=optimized_rop,
            improvement_pct=improvement,
            recommended=recommended,
            baseline_params={p: float(conditions[p]) for p in PARAM_BOUNDS},
            risk_before=risk_before,
            risk_after=risk_after,
            trade_offs=trade_offs,
        )

    @staticmethod
    def _build_trade_offs(conditions, recommended, risk_before, risk_after) -> list[str]:
        msgs = []
        for p in PARAM_BOUNDS:
            cur, rec = conditions[p], recommended[p]
            if abs(rec - cur) / max(abs(cur), 1e-6) > 0.05:
                direction = "increase" if rec > cur else "reduce"
                msgs.append(f"{direction.capitalize()} {p.replace('_', ' ')} "
                            f"from {cur:.0f} to {rec:.0f}")
        dv = risk_after["vibration_risk"] - risk_before["vibration_risk"]
        if dv < -0.02:
            msgs.append(f"Vibration risk reduced by {abs(dv)*100:.0f}%")
        elif dv > 0.02:
            msgs.append(f"Caution: vibration risk rises {dv*100:.0f}% — monitor MSE")
        db = risk_after["bit_damage_risk"] - risk_before["bit_damage_risk"]
        if db > 0.02:
            msgs.append(f"Caution: bit-damage risk rises {db*100:.0f}% — inspect bit on next trip")
        return msgs or ["Current parameters are already near-optimal"]
