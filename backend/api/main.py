"""FastAPI application — Oil & Gas Drilling ROP Prediction & Optimization API.

Endpoints:
  GET  /health
  POST /predict/rop
  POST /predict/efficiency
  POST /predict/risk
  POST /optimize/drilling
  GET  /dashboard/summary
  GET  /well/{well_id}
  POST /explain-prediction
  GET  /risk/overview
  POST /export/optimization-report   (PDF)
  GET  /export/predictions.csv       (CSV)
  POST /upload/score                 (batch score uploaded CSV)
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from backend.api.schemas import (
    DrillingConditions, ROPResponse, EfficiencyResponse, RiskResponse,
    OptimizeResponse, ExplainResponse, HealthResponse,
)
from backend.services import model_registry as reg
from backend.services import dashboard_service as dash
from backend.services import report_service as report
from backend.services.database import init_db
from backend.utils.features import add_engineered_features
from backend.utils.logger import get_logger

log = get_logger("api")

VERSION = "1.0.0"

app = FastAPI(
    title="Drilling ROP Prediction & Optimization API",
    description="AI-powered drilling analytics: ROP prediction, risk scoring, "
                "parameter optimization, and explainable AI.",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    try:
        init_db()
    except Exception as e:  # pragma: no cover
        log.warning("DB init at startup failed: %s", e)


def _guard(fn):
    """Translate ModelNotTrainedError into a clean 503."""
    try:
        return fn()
    except reg.ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))


# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return HealthResponse(status="ok", models_available=reg.models_available(), version=VERSION)


@app.post("/predict/rop", response_model=ROPResponse, tags=["prediction"])
def predict_rop(payload: DrillingConditions):
    def _run():
        model = reg.get_rop_model()
        df = add_engineered_features(pd.DataFrame([payload.to_features()]))
        pred = float(model.predict(df)[0])
        return ROPResponse(predicted_rop=round(pred, 2), model=getattr(model, "name", "rop"))
    return _guard(_run)


@app.post("/predict/efficiency", response_model=EfficiencyResponse, tags=["prediction"])
def predict_efficiency(payload: DrillingConditions):
    def _run():
        risk = reg.get_risk_model()
        df = add_engineered_features(pd.DataFrame([payload.to_features()]))
        score = float(risk.predict(df)["drilling_efficiency_score"][0])
        rating = ("Excellent" if score > 0.75 else "Good" if score > 0.55
                  else "Fair" if score > 0.35 else "Poor")
        return EfficiencyResponse(drilling_efficiency_score=round(score, 3), rating=rating)
    return _guard(_run)


@app.post("/predict/risk", response_model=RiskResponse, tags=["prediction"])
def predict_risk(payload: DrillingConditions):
    def _run():
        risk = reg.get_risk_model()
        df = add_engineered_features(pd.DataFrame([payload.to_features()]))
        p = risk.predict(df)
        vib = float(p["vibration_risk"][0])
        bit = float(p["bit_damage_risk"][0])
        stuck = float(p["stuck_pipe_risk"][0])
        idx = float(np.mean([vib, bit, stuck]))
        level = "High" if idx > 0.6 else "Medium" if idx > 0.35 else "Low"
        return RiskResponse(vibration_risk=round(vib, 3), bit_damage_risk=round(bit, 3),
                            stuck_pipe_risk=round(stuck, 3),
                            instability_index=round(idx, 3), risk_level=level)
    return _guard(_run)


@app.post("/optimize/drilling", response_model=OptimizeResponse, tags=["optimization"])
def optimize_drilling(payload: DrillingConditions):
    def _run():
        optimizer = reg.get_optimizer()
        result = optimizer.optimize(payload.to_features()).to_dict()
        return OptimizeResponse(**result)
    return _guard(_run)


@app.post("/explain-prediction", response_model=ExplainResponse, tags=["explainability"])
def explain_prediction(payload: DrillingConditions):
    def _run():
        explainer = reg.get_explainer()
        return ExplainResponse(**explainer.explain(payload.to_features()))
    return _guard(_run)


@app.get("/dashboard/summary", tags=["dashboard"])
def dashboard_summary():
    return dash.executive_summary()


@app.get("/well/{well_id}", tags=["dashboard"])
def get_well(well_id: str):
    data = dash.well_detail(well_id)
    if not data.get("available"):
        raise HTTPException(status_code=404, detail=data.get("message", "Well not found"))
    return data


@app.get("/risk/overview", tags=["dashboard"])
def risk_overview():
    return dash.risk_overview()


# ---- exports ----
@app.post("/export/optimization-report", tags=["export"])
def export_optimization_report(payload: DrillingConditions):
    def _run():
        optimizer = reg.get_optimizer()
        result = optimizer.optimize(payload.to_features()).to_dict()
        pdf = report.optimization_to_pdf(result)
        return Response(content=pdf, media_type="application/pdf",
                        headers={"Content-Disposition": "attachment; filename=optimization_report.pdf"})
    return _guard(_run)


@app.get("/export/predictions.csv", tags=["export"])
def export_predictions_csv():
    from backend.services.database import query_df, table_exists
    if not table_exists("predictions"):
        raise HTTPException(status_code=404, detail="No predictions available")
    df = query_df("SELECT * FROM predictions")
    data = report.dataframe_to_csv_bytes(df)
    return StreamingResponse(io.BytesIO(data), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})


@app.post("/upload/score", tags=["batch"])
async def upload_and_score(file: UploadFile = File(...)):
    """Score an uploaded drilling-report CSV (batch, no streaming)."""
    model = _guard(lambda: reg.get_rop_model())
    risk = _guard(lambda: reg.get_risk_model())

    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    df = add_engineered_features(df)
    try:
        df["pred_rop"] = model.predict(df)
        for t, vals in risk.predict(df).items():
            df[f"pred_{t}"] = vals
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scoring failed (check columns): {e}")

    return {
        "rows_scored": int(len(df)),
        "avg_pred_rop": round(float(df["pred_rop"].mean()), 2),
        "preview": df.head(20).round(3).to_dict(orient="records"),
    }
