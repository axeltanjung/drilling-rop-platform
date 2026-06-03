"""Explainable AI module using SHAP.

Provides:
  * global feature importance (mean |SHAP|)
  * per-prediction local explanations (contribution of each drilling factor)
  * prediction confidence proxy from tree-ensemble spread
  * human-readable operational explanation of why ROP increased/decreased
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.utils.features import add_engineered_features
from backend.utils.logger import get_logger

log = get_logger("explainer")

try:
    import shap
    _HAS_SHAP = True
except Exception:  # pragma: no cover
    _HAS_SHAP = False
    log.warning("shap not installed; falling back to model feature importances")


# Friendly names for operational narrative
FRIENDLY = {
    "weight_on_bit": "Weight on Bit",
    "rpm": "RPM",
    "torque": "Torque",
    "mud_flow_rate": "Mud Flow Rate",
    "bit_wear": "Bit Wear",
    "vibration_level": "Vibration",
    "depth": "Depth",
    "formation_type": "Formation",
    "bit_type": "Bit Type",
    "mechanical_specific_energy": "Mechanical Specific Energy",
    "wob_rpm_product": "WOB x RPM",
    "hydraulic_power": "Hydraulic Power",
}


class ROPExplainer:
    def __init__(self, rop_model):
        self.rop_model = rop_model
        self.features = getattr(rop_model, "features", [])
        self._shap = None
        if _HAS_SHAP and hasattr(rop_model, "model"):
            try:
                self._shap = shap.TreeExplainer(rop_model.model)
            except Exception as e:  # deep models won't be tree-based
                log.warning("TreeExplainer init failed: %s", e)

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c in getattr(self.rop_model, "cat_features", []):
            X[c] = X[c].astype("category")
        return X

    def global_importance(self, sample: pd.DataFrame, top: int = 12) -> list[dict]:
        sample = add_engineered_features(sample)
        if self._shap is not None:
            X = self._prep(sample)
            vals = self._shap.shap_values(X)
            mean_abs = np.abs(vals).mean(axis=0)
            pairs = sorted(zip(self.features, mean_abs.tolist()), key=lambda x: -x[1])
        else:
            fi = self.rop_model.feature_importance()
            pairs = list(fi.items())
        total = sum(v for _, v in pairs) or 1.0
        return [
            {"feature": f, "label": FRIENDLY.get(f, f), "importance": round(v / total, 4)}
            for f, v in pairs[:top]
        ]

    def explain(self, conditions: dict, top: int = 8) -> dict:
        """Local explanation for a single prediction."""
        row = add_engineered_features(pd.DataFrame([conditions]))
        pred = float(self.rop_model.predict(row)[0])

        contributions = []
        confidence = None
        if self._shap is not None:
            X = self._prep(row)
            vals = self._shap.shap_values(X)[0]
            base = float(np.array(self._shap.expected_value).ravel()[0])
            order = np.argsort(-np.abs(vals))
            for i in order[:top]:
                f = self.features[i]
                contributions.append({
                    "feature": f,
                    "label": FRIENDLY.get(f, f),
                    "value": _fmt_value(conditions.get(f)),
                    "contribution": round(float(vals[i]), 3),
                    "direction": "increases" if vals[i] > 0 else "decreases",
                })
            confidence = self._confidence(row)
            base_value = base
        else:
            fi = self.rop_model.feature_importance()
            base_value = pred
            for f, imp in list(fi.items())[:top]:
                contributions.append({
                    "feature": f, "label": FRIENDLY.get(f, f),
                    "value": _fmt_value(conditions.get(f)),
                    "contribution": round(float(imp), 3), "direction": "influences",
                })

        narrative = self._narrative(pred, contributions)
        return {
            "predicted_rop": round(pred, 2),
            "base_value": round(base_value, 2),
            "confidence": confidence,
            "contributions": contributions,
            "narrative": narrative,
        }

    def _confidence(self, row: pd.DataFrame) -> dict | None:
        """Confidence proxy from per-tree prediction spread (LightGBM)."""
        try:
            booster = self.rop_model.model.booster_
            X = self._prep(row)
            n = booster.num_trees()
            step = max(1, n // 30)
            preds = [booster.predict(X, num_iteration=i)[0] for i in range(step, n + 1, step)]
            std = float(np.std(preds))
            mean = float(np.mean(preds))
            return {"std": round(std, 3),
                    "interval_low": round(mean - 1.96 * std, 2),
                    "interval_high": round(mean + 1.96 * std, 2),
                    "level": "high" if std < 2 else "medium" if std < 5 else "low"}
        except Exception:
            return None

    @staticmethod
    def _narrative(pred: float, contributions: list[dict]) -> str:
        if not contributions:
            return f"Predicted ROP is {pred:.1f} ft/hr."
        up = [c for c in contributions if c["direction"] == "increases"][:2]
        down = [c for c in contributions if c["direction"] == "decreases"][:2]
        parts = [f"Predicted ROP is {pred:.1f} ft/hr."]
        if up:
            parts.append("Boosted mainly by " + ", ".join(c["label"] for c in up) + ".")
        if down:
            parts.append("Held back by " + ", ".join(c["label"] for c in down) + ".")
        return " ".join(parts)


def _fmt_value(v):
    if isinstance(v, (int, float, np.floating, np.integer)):
        return round(float(v), 2)
    return v
