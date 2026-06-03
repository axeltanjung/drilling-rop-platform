\# System Architecture



\## Overview

The platform is a 3-tier system: \*\*React dashboard → FastAPI service → ML models + SQLite + MLflow\*\*.

All components run locally and use \*\*batch\*\* datasets (no streaming).



\## Components



\### 1. Data Layer

\- `backend/data/synthetic\_drilling\_data\_generator.py` — physics-inspired synthetic telemetry (>=200k rows).

\- `data/raw/drilling\_telemetry.csv` — generated dataset.

\- `data/drilling.db` — SQLite store of batch predictions (`predictions`, `well\_summary` tables).



\### 2. ML Layer (`backend/models`, `backend/training`)

| Model | File | Role |

|---|---|---|

| LightGBM | `models/lightgbm\_model.py` | Primary ROP regressor + SHAP |

| CatBoost | `models/catboost\_model.py` | Benchmark + interactions |

| Deep (MLP/LSTM) | `models/deep\_model.py` | Non-linear / sequential |

| Risk/Efficiency | `models/risk\_model.py` | Multi-target risk + efficiency |



Pipeline: `training/preprocess.py` → `training/train\_pipeline.py` → MLflow (`training/mlflow\_tracking.py`) → `training/batch\_predict.py`.



\### 3. Intelligence Layer

\- `optimization/optimizer.py` — surrogate-model parameter search (WOB/RPM/MudFlow) maximizing penalized ROP.

\- `explainability/explainer.py` — SHAP local/global explanations + confidence proxy.



\### 4. Service Layer (`backend/services`, `backend/api`)

\- `services/database.py` — SQLAlchemy engine.

\- `services/model\_registry.py` — cached model loaders.

\- `services/dashboard\_service.py` — aggregations for the dashboard.

\- `services/report\_service.py` — PDF/CSV export.

\- `api/main.py` — FastAPI app exposing all endpoints + Swagger.



\### 5. Presentation Layer (`frontend`)

\- React + Vite + TailwindCSS + Recharts + Framer Motion.

\- 5 pages: Overview, Well Detail, AI Insights, Optimization, Risk Analysis.



\## Request Flow (prediction)

```

Dashboard form → POST /predict/rop → model\_registry.get\_rop\_model()

&#x20;  → add\_engineered\_features() → model.predict() → JSON → chart

```



\## Deployment

`docker-compose.yml` builds three services: `backend` (uvicorn), `frontend` (nginx serving built SPA + /api proxy), and `mlflow` (tracking UI).

The backend entrypoint auto-bootstraps data + a fast model train on first run if no artifacts exist.



