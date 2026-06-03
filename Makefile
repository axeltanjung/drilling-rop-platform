.PHONY: install data train predict serve frontend test docker clean

install:
	pip install -r backend/requirements.txt

data:
	python backend/data/synthetic_drilling_data_generator.py

train:
	python backend/training/train_pipeline.py

train-fast:
	python backend/training/train_pipeline.py --fast --no-tune

predict:
	python backend/training/batch_predict.py

serve:
	uvicorn backend.api.main:app --reload --port 8000

mlflow:
	mlflow ui --backend-store-uri ./mlflow/mlruns --port 5000

frontend:
	cd frontend && npm install && npm run dev

test:
	pytest -q tests

docker:
	docker-compose up --build

bootstrap: data train-fast predict
	@echo "Bootstrap complete. Run 'make serve' and 'make frontend'."

clean:
	rm -rf data/raw/*.csv data/processed/*.csv data/*.db backend/models/artifacts/* mlflow/mlruns/*
