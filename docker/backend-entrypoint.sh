#!/usr/bin/env bash
set -e

cd /app

echo "[entrypoint] Checking for trained models..."
if [ ! -f "backend/models/artifacts/lightgbm_rop.pkl" ]; then
  echo "[entrypoint] No models found. Bootstrapping (data + fast training)..."
  python backend/data/synthetic_drilling_data_generator.py
  python backend/training/train_pipeline.py --fast --no-tune
  python backend/training/batch_predict.py
else
  echo "[entrypoint] Models found. Skipping bootstrap."
fi

echo "[entrypoint] Starting API on :8000"
exec uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
