PYTHON ?= python

.PHONY: all lint test build-dataset extract analyze evaluate docker-build docker-up

all: lint test build-dataset extract analyze evaluate

lint:
	ruff check src tests scripts
	black --check src tests scripts

test:
	pytest -q

build-dataset:
	$(PYTHON) -m speechlab.cli build_dataset

extract:
	$(PYTHON) -m speechlab.cli extract_features

analyze:
	$(PYTHON) -m speechlab.cli analyze

evaluate:
	$(PYTHON) -m speechlab.cli evaluate

docker-build:
	docker compose build

docker-up:
	docker compose up -d
