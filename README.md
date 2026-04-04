# Hybrid Multi-Basket FX Forecasting Pipeline

## Project Overview
This repository provides the architectural framework for a 3rd-year Time Series Analysis final project. It replicates and modernizes the methodology proposed by **Ince & Trafalis (2006)**.

The implementation supports a dynamic, target-centric FX basket (currently focused on **PHP**: USD/PHP, CNY/PHP, JPY/PHP, HKD/PHP, SGD/PHP) using daily liquidity data from **2010 to 2026**.

---

### CRITICAL FOR TEAM MEMBERS
DO NOT ALTER THE CORE TEMPORAL LOGIC. This pipeline adheres to a strict chronological evaluation framework.
1. **Always consult `configs/pipeline_config.yaml`**: The pipeline is modular; all thresholds, targets, and parameters should be set there first.
2. **Modular Integrity**: When adding/modifying code, ensure it remains modular and does not break existing dependencies.
3. **Preserve unrelated code/comments**: Do not delete blocks or notes unless explicitly required for the task.
4. **No future-data leakage**: One-step-ahead forecasting and sequential splitting (80/10/10) must be maintained for academic validity.

---

## Architecture & Data Structure
The pipeline follows a **Group-by-Currency** architecture.

### Directory Structure:
* `src/`: Core Python/R processing scripts.
* `data/raw/{CCY}/`: Raw data fetched from external sources.
* `data/processed/{CCY}/`: Aligned, cleaned, and transformed CSVs (`fx_aligned.csv`, `train.csv`, etc.).
* `results/{CCY}/`: Forecast outputs, evaluation reports, and diagnostic artifacts.
* `notebooks/`: For visual EDA and descriptive diagnostics (e.g., `01_descriptive_statistics.ipynb`).

### Data Flow:
1. **01_data_loader.py**: Fetches data, executes cleaning, and creates chronological splits.
2. **02_diagnostics.R**: Econometric stationarity and structural break tests.
3. **03_parametric.R**: ARIMA/VAR linear baselines. 
4. **04_nonparametric.py**: SVR/MLP non-linear correction stage (Hybrid Stage).
5. **05_evaluation.py**: Metrics calculation and Diebold-Mariano statistical testing.

---

## Setup & Execution

**Prerequisites:** 
* Python 3.10+ and standard `pip`.
* R 4.5.x with libraries: `forecast`, `vars`, `strucchange`, `jsonlite`, `urca`.
* `.env` file containing optional `FRED_API_KEY` for interest rate data (placeholder used if missing).

### Installation
```bash
pip install -r requirements.txt
Rscript scripts/setup_r_env.R
```

### Execution
The `main.py` is the primary orchestrator. Note that data and diagnostics are pre-processed and included in the repo by default.

**Default Routine (Standard Evaluation):**
```bash
python main.py --skip-data --skip-diagnostics
```

**Experimental HPO:**
Do **NOT** run the Hyperparameter Optimization (`--run-hpo`) for general evaluation tasks. It is extremely time-intensive and only intended for targeted tuning phases.

**Modular Scripts:**
You can run any script individually for debugging:
`python src/01_data_loader.py --config configs/pipeline_config.yaml`

---

## Output Structure (Results)
Results are nested by the `active_target` specified in the config:
* `results/{CCY}/arima/`: Univariate linear results.
* `results/{CCY}/var/`: Multivariate linear results.
* `results/{CCY}/evaluation/`: Metrics, DM tests, and final forecasting plots.
* `results/optuna_hpo.db`: Persistent ML study database (if HPO was run).
