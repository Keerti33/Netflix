# Netflix Prize Recommendation System — Makefile
# Usage:  make all          (run full pipeline end-to-end)
#         make <target>     (run a specific step)
#
# On Windows without GNU Make, run each command manually in order.

PYTHON   ?= python
PIP      ?= pip
SHELL    := cmd

# -------------------------------------------------------------------------
# Targets
# -------------------------------------------------------------------------

.PHONY: install ingest sample preprocess eda train evaluate recommend analysis all clean

## Install Python dependencies
install:
	$(PIP) install -r requirements.txt

## Step 1: Ingest raw Netflix Prize data -> data/processed/ratings.parquet
ingest:
	$(PYTHON) src/data/ingest.py

## Step 2: Filter & sample ratings -> data/processed/ratings_sample.parquet
sample:
	$(PYTHON) src/data/sample.py --sample_size 200000 --min_user_ratings 50 --min_movie_ratings 100

## Step 3: Preprocess (split, remap, CSR, implicit matrix)
preprocess:
	$(PYTHON) src/data/preprocess.py

## Step 4: EDA notebook (launch Jupyter — run manually)
eda:
	@echo "Launch Jupyter and open notebooks/01_eda.ipynb"

## Step 5: Train all models (baselines + MF)
train:
	$(PYTHON) src/models/run_baselines.py
	$(PYTHON) src/models/run_mf.py --no_tune

## Step 6: Evaluate all models
evaluate:
	$(PYTHON) src/evaluation/evaluate_all.py

## Step 7: Generate recommendations for sample users
recommend:
	$(PYTHON) src/recommend/generate.py --n_users 500

## Step 8: Run success/failure analysis
analysis:
	$(PYTHON) src/recommend/analysis.py

## Step 9: Run unit tests
test:
	$(PYTHON) -m pytest tests/ -v

## Full pipeline (end-to-end)
all: install ingest sample preprocess train evaluate recommend analysis test
	@echo "=== Full pipeline complete ==="

## Remove generated outputs (keep raw data)
clean:
	@echo Cleaning outputs...
	-rmdir /s /q outputs\\models 2>nul
	-rmdir /s /q outputs\\predictions 2>nul
	-rmdir /s /q outputs\\results 2>nul
	-rmdir /s /q outputs\\recommendations 2>nul
	-rmdir /s /q outputs\\analysis 2>nul
	@echo Done.
