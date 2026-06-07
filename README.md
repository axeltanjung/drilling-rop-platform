# 🛢️ Oil & Gas Drilling ROP Prediction & Optimization Platform

> An AI-powered, enterprise-grade drilling analytics platform that predicts **Rate of Penetration (ROP)**, scores **drilling efficiency**, detects **performance degradation**, and recommends **optimal drilling parameters** — built to reduce drilling cost and non-productive time (NPT).

![status](https://img.shields.io/badge/status-portfolio--grade-success)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents
1. [Project Background](#1-project-background)
2. [Business Benefits](#2-business-benefits)
3. [Domain Knowledge](#3-domain-knowledge)
4. [AI/ML Architecture](#4-aiml-architecture)
5. [System Architecture](#5-system-architecture)
6. [Deployment Guide](#6-deployment-guide)
7. [Future Improvements](#7-future-improvements)

---
## Project Preview

<img src="assets/AI Insight.png" width="1200"/>

<img src="assets/Drilling Optimization.png" width="1200"/>

## 1. Project Background

### What is ROP?
**Rate of Penetration (ROP)** measures how fast a drill bit penetrates rock formations, typically expressed in **feet per hour (ft/hr)** or **meters per hour (m/hr)**. It is one of the single most important Key Performance Indicators (KPIs) in drilling operations because it directly drives the time — and therefore the cost — required to drill a well.

### Why Drilling Optimization Matters
Offshore and onshore drilling rigs can cost **$30,000 – $1,000,000+ per day** to operate. Every hour saved through better drilling parameter selection translates directly into substantial cost reduction. ROP directly affects:

- **Drilling duration** — faster ROP means fewer rig days
- **Rig operational cost** — day-rate × number of days
- **Fuel usage** — diesel consumption on the rig
- **Bit wear** — aggressive parameters wear bits faster, forcing costly bit trips
- **Non-Productive Time (NPT)** — drilling dysfunctions cause downtime

### Operational Challenges
Selecting drilling parameters (Weight-on-Bit, RPM, mud flow) is a **multi-objective trade-off**. Poor selection can lead to:
- Slow drilling and wasted rig time
- Excessive vibration (stick-slip, bit bounce, whirl)
- Bit damage and premature bit trips
- Stuck pipe incidents (one of the most expensive NPT events)
- Increased operational cost and HSE risk

### Why AI Is Useful
Drilling generates rich **multivariate, non-linear, time-series telemetry**. The relationships between WOB, RPM, formation hardness, hydraulics, and ROP are complex and formation-dependent. Machine learning models can:
- Learn non-linear physics-inspired relationships from historical data
- Predict ROP and risk in real operating conditions
- Recommend parameter changes that maximize ROP while minimizing risk
- Explain *why* a prediction was made (critical for engineer trust)

---

## 2. Business Benefits

| Benefit | Impact |
|---|---|
| **Reduced drilling cost** | Fewer rig days at high day-rates |
| **Faster drilling** | Higher sustained ROP across formations |
| **Improved bit life** | Avoid aggressive parameter combinations that destroy bits |
| **Lower NPT** | Predict and avoid stuck-pipe / vibration events |
| **Operational efficiency** | Data-driven, repeatable parameter selection |
| **Reduced equipment stress** | Lower torque and vibration extends tool life |

---

## 3. Domain Knowledge

### The Drilling Process
A rotating drill bit, suspended on a drill string, crushes and cuts rock. Drilling fluid (**mud**) is pumped down the drill string, exits the bit nozzles, and circulates rock cuttings back to surface while cooling the bit and stabilizing the wellbore.

### Key Parameters
- **Weight on Bit (WOB)** — the downward force applied to the bit (klbs). Higher WOB generally increases ROP, but excessive WOB causes vibration and bit damage.
- **RPM (Revolutions Per Minute)** — bit rotation speed. Higher RPM can increase ROP but also raises vibration risk and bit wear.
- **Mud Flow Rate (GPM)** — controls hole cleaning and bit cooling. Inadequate flow leaves cuttings in the hole, reducing efficiency and risking stuck pipe.
- **Torque** — rotational resistance. Fluctuating torque indicates drilling instability (stick-slip).
- **Standpipe / Pump Pressure** — hydraulic pressure of the circulating system.
- **Hook Load** — weight measured at the hook supporting the drill string.

### Bit Wear & Formation Hardness
- **Bit wear** accumulates with drilling hours and aggressive parameters, gradually reducing ROP and efficiency.
- **Formation hardness** is the dominant driver of ROP — soft formations (sandstone) drill fast; hard formations (granite, dolomite) drill slowly.

### Drilling Dysfunction
- **Stick-slip** — torsional vibration (bit stalls then snaps free)
- **Bit bounce / whirl** — axial and lateral vibration
- **Stuck pipe** — drill string becomes immovable in the hole
- **Inadequate hole cleaning** — cuttings accumulate

---

## 4. AI/ML Architecture

The platform trains and benchmarks **three complementary models**:

| Model | Type | Purpose |
|---|---|---|
| **LightGBM** | Gradient-boosted trees | Primary tabular ROP predictor with SHAP explainability |
| **CatBoost** | Gradient-boosted trees | Native categorical handling + benchmark |
| **Deep Learning (MLP / LSTM)** | Neural network | Captures non-linear + sequential drilling behavior |

### Pipeline Stages
1. **Data preprocessing** — cleaning, imputation, outlier handling
2. **Feature engineering** — physics-inspired interaction features
3. **Sequence generation** — sliding windows for LSTM
4. **Model training** — three model families
5. **Hyperparameter tuning** — Optuna / grid search
6. **Model evaluation** — RMSE, MAE, R²
7. **MLflow tracking** — metrics, params, artifacts
8. **Model registry** — versioned model promotion
9. **Batch prediction pipeline** — scoring uploaded drilling reports

### Optimization Engine
A surrogate-model-based optimizer searches the (WOB, RPM, Mud Flow) space to **maximize predicted ROP** subject to **vibration and bit-wear penalties**, returning recommended settings and estimated performance gain.

### Explainable AI
**SHAP** values explain every prediction — which drilling variables increased or decreased ROP — plus global feature importance and prediction confidence intervals.

---

## 5. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        React + Vite Frontend                   │
│   Executive · Well Detail · AI Insights · Optimize · Risk      │
│         TailwindCSS · Plotly/Recharts · Framer Motion          │
└───────────────────────────────┬──────────────────────────────┘
                                 │ REST (Axios)
┌───────────────────────────────▼──────────────────────────────┐
│                         FastAPI Backend                        │
│  /predict/* · /optimize/drilling · /explain-prediction · ...   │
│   Services · Optimization · Explainability · Utils             │
└───────┬───────────────────────┬───────────────────────┬──────┘
        │                       │                        │
┌───────▼──────┐    ┌───────────▼─────────┐    ┌─────────▼──────┐
│  SQLite DB   │    │  Model Artifacts     │    │     MLflow      │
│  wells/preds │    │ LightGBM/CatBoost/DL │    │ tracking+reg.   │
└──────────────┘    └─────────────────────┘    └────────────────┘
```

### ML Pipeline Flow
```
synthetic_drilling_data_generator.py
        │ (CSV: data/raw/drilling_telemetry.csv)
        ▼
preprocess → feature_engineering → sequence_gen
        ▼
train (LightGBM | CatBoost | DL) → MLflow → model registry
        ▼
batch_predict.py → SQLite predictions → API → Dashboard
```

---

## 6. Deployment Guide

### Local Setup (Python)
```bash
# 1. Clone and enter
cd drilling-rop-platform

# 2. Backend env
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# 3. Generate synthetic data (200k+ rows)
python backend/data/synthetic_drilling_data_generator.py

# 4. Train models (logs to MLflow)
python backend/training/train_pipeline.py

# 5. Run batch predictions + seed SQLite
python backend/training/batch_predict.py

# 6. Start API
uvicorn backend.api.main:app --reload --port 8000
# Swagger docs: http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Dashboard: http://localhost:5173
```

### MLflow UI
```bash
mlflow ui --backend-store-uri ./mlflow/mlruns --port 5000
# http://localhost:5000
```

### Docker (one command)
```bash
docker-compose up --build
# Frontend: http://localhost:5173  (or :80 via nginx)
# Backend:  http://localhost:8000/docs
# MLflow:   http://localhost:5000
```

### Environment Variables
See `.env.example`:
```
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=sqlite:///./data/drilling.db
MLFLOW_TRACKING_URI=./mlflow/mlruns
MODEL_DIR=backend/models/artifacts
VITE_API_BASE_URL=http://localhost:8000
```

### API Documentation
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check |
| `/predict/rop` | POST | Predict Rate of Penetration |
| `/predict/efficiency` | POST | Predict drilling efficiency score |
| `/predict/risk` | POST | Predict stuck-pipe / vibration / bit-damage risk |
| `/optimize/drilling` | POST | Recommend optimal WOB / RPM / Mud Flow |
| `/dashboard/summary` | GET | Executive KPI summary |
| `/well/{id}` | GET | Per-well time-series detail |
| `/explain-prediction` | POST | SHAP explanation for a prediction |

---

## 7. Future Improvements
- **Reinforcement Learning** drilling optimization (closed-loop autopilot)
- **Real-time telemetry integration** (WITSML / OPC-UA ingestion)
- **Digital twin** drilling simulation
- **Edge deployment** on rig-site hardware
- **Autonomous drilling assistant** with LLM-based operator guidance

---

## Tech Stack
**Backend:** Python · FastAPI · Pandas · NumPy · Scikit-learn · LightGBM · CatBoost · TensorFlow/PyTorch · MLflow
**Frontend:** React + Vite · TailwindCSS · Plotly/Recharts · Framer Motion · Axios
**Database:** SQLite · **Deployment:** Docker · docker-compose · Nginx

## Project Structure
```
drilling-rop-platform/
├── backend/
│   ├── api/              # FastAPI app, routers, schemas
│   ├── models/           # model wrappers + artifacts
│   ├── training/         # training & batch prediction pipeline
│   ├── optimization/     # drilling parameter optimizer
│   ├── explainability/   # SHAP explainers
│   ├── services/         # business logic, DB access
│   ├── data/             # synthetic data generator
│   └── utils/            # config, logging, features
├── frontend/             # React + Vite dashboard
├── data/                 # generated datasets + SQLite db
├── notebooks/            # EDA notebooks
├── mlflow/               # MLflow tracking store
├── docs/                 # documentation + dataset docs
├── screenshots/          # dashboard screenshots
├── tests/                # pytest suite
├── docker/               # Dockerfiles + nginx
└── docker-compose.yml
```

> ⚠️ All data is **synthetic** and physics-inspired for demonstration. Not for operational drilling decisions.
