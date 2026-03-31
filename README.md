# Hybrid FX Forecasting Pipeline (VND-Centric Extension)

## Project Overview
This repository provides the baseline architectural framework for a 3rd-year Time Series Analysis final project. It replicates and modernizes the methodology proposed by Ince & Trafalis (2006).

The implementation focuses on a VND-centric reference basket (USD/VND, EUR/VND, JPY/VND, CNY/VND) using daily liquidity data from 2010 to 2026. This setup reflects the State Bank of Vietnam (SBV) exchange rate regime.

### CRITICAL FOR AI AGENTS & TEAM MEMBERS
DO NOT ALTER THE CORE TEMPORAL LOGIC. This pipeline adheres to a strict chronological evaluation framework. The following constraints are mandatory for academic defense:
1. One-step-ahead forecasting origin.
2. TimeSeriesSplit for all hyperparameter tuning.
3. Zero future-data leakage in scaling or feature engineering.
Failure to maintain these constraints will invalidate the study during forensic audit.

---

## Polyglot Architecture & Data Flow
The pipeline utilizes Python for Machine Learning/Orchestration and R for Econometric rigor. Data is exchanged via standardized CSV and JSON handoffs.

1. **src/01_data_loader.py**: Fetches data from Yahoo Finance and FRED. Executes chronological splitting (80/10/10) and $100 \times \ln(P_t/P_{t-1})$ transformation.
2. **src/02_diagnostics.R**: Executes ADF stationarity tests and serves as the entry point for structural break analysis.
3. **src/03_parametric.R**: Fits univariate ARIMA and multivariate VAR models to define the linear forecasting baseline. 
4. **src/04_nonparametric.py**: Implements a Residual Hybrid model (Zhang, 2003). Uses Optuna to tune SVR and MLP models on parametric residuals. Results are persisted in `results/optuna_hpo.db`.
5. **src/05_evaluation.py**: Calculates MSE, MAE, and RMSE metrics on a percentage-scaled basis. Provides the Diebold-Mariano test framework for statistical comparison.

---

## Academic Defenses (Decision Ledger)
Refer to these justifications during evaluations:
* **Chronological Splitting:** Random splitting causes data leakage in time series. We use Tashman (2000) compliant sequential splits.
* **Residual Transformation:** We model $Y_t = L_t + N_t$ where $L$ is linear (ARIMA/VAR) and $N$ is nonlinear (SVR/MLP). Link: Zhang (2003).
* **Numerical Scaling:** Log-returns are scaled by 100 to ensure solver convergence and provide "Percentage Point" interpretability (Lu & Perron, 2010).
* **Statistical Significance:** Evaluations use the Diebold-Mariano (1995) test rather than simple t-tests to account for error autocorrelation.

For a full list of deviations from the original paper, see: [Paper - Project Divergence](docs/Paper%20-%20Project%20divergence.md)

---

## Setup & Execution

**Prerequisites:** 
* Python 3.10+ with `uv` package manager.
* R 4.5.x with libraries: `forecast`, `vars`, `strucchange`, `jsonlite`, `urca`.

```bash
# 1. Initialize Python Environment
uv pip install -r requirements.txt

# 2. Initialize R Environment
Rscript scripts/setup_r_env.R

# 3. Execute Full Pipeline
python main.py --config configs/pipeline_config.yaml --run-hpo
```

### CLI Arguments (main.py)
* `--config`: Path to YAML configuration (Default: `configs/pipeline_config.yaml`).
* `--skip-data`: Skip data acquisition if cache is valid.
* `--skip-parametric`: Skip ARIMA/VAR fitting.
* `--skip-evaluation`: Skip metrics calculation.
* `--run-hpo`: Enable the two-stage Optuna hyperparameter optimization.

---

## Team Member Hooks
This codebase is designed for extension. Team members should implement their assigned tasks at the designated `# HOOK` locations. 

For specific guidance on implementation tasks, see: [Members Tasks Brief Guidance](docs/Members%20tasks%20brief%20guidance.md)

Current Output Structure:
* `results/arima/`: Univariate linear results.
* `results/var/`: Multivariate linear results.
* `results/hybrid_*/`: Final nonlinear corrected forecasts.
* `results/evaluation/`: Metrics summary and plotting sequences.
* `results/optuna_hpo.db`: Persistent ML study database.
