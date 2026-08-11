.PHONY: install format lint test train predict score report docker-build api-docker-build

install:
	python -m pip install -r requirements.txt

format:
	python -m black --check .

lint:
	python -m ruff check .

test:
	python -m pytest -q

train:
	python src/train.py

predict:
	python src/predict.py

score:
	python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_predictions.csv --output-dir outputs/scorer_results

report:
	python scripts/create_report.py

docker-build:
	docker build -t spotter-assessment:local .

api-docker-build:
	docker build -f api/Dockerfile -t spotter-api:local .
