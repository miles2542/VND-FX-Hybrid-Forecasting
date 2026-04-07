# VND-FX-Hybrid-Forecasting

This repository contains a configurable FX forecasting pipeline with linear (ARIMA/VAR), hybrid residual-learning models (SVR/MLP), and evaluation outputs.

## Active Target

The active currency target is controlled in `configs/pipeline_config.yaml` via:

- `active_target: PHP`

All new notebooks below auto-read this config and should be rerun after changing `active_target`.

## Analysis Pipeline Flow

Time-Series Diagnostics -> Model Justification -> Statistical Validation -> Robustness -> Interpretation

1. `khanh_model_analysis/00_comprehensive_time_series_analysis.ipynb`
- Deep time-series diagnostics: stationarity (ADF/KPSS/PP), structural breaks, Johansen cointegration, Granger matrix, rolling diagnostics, lead-lag and spillover visualizations.
- Exports diagnostics tables and interactive figures to `results/<target>/diagnostics/`.

2. `khanh_model_analysis/01_statistical_tests.ipynb`
- ARIMA/VAR parameter significance.
- Diebold-Mariano (DM) tests and residual diagnostics.
- Exports to `results/<target>/evaluation/`.

3. `khanh_model_analysis/02_robustness_checks.ipynb`
- RESET specification-bias checks.
- Validation/test stability and DM tail-sensitivity checks.

4. `khanh_model_analysis/03_results_interpretation.ipynb`
- Publication tables, DM summary, TS diagnostics -> model-choice linkage.
- Forecast/error visualization and LaTeX table export.

5. `khanh_model_analysis/05_forecast_visualization_and_economic_context.ipynb` (optional)
- Interactive forecast dashboard, cumulative error paths, event-overlay template.

## New Statistical Validation and Interpretation Notebooks

The following core notebooks are under `khanh_model_analysis/`:

0. `khanh_model_analysis/00_comprehensive_time_series_analysis.ipynb`
- Rich diagnostics for stationarity, breakpoints, cointegration, Granger causality, and volatility spillover.
- Saves publication-ready tables/plots to `results/<target>/diagnostics/`.

1. `khanh_model_analysis/01_statistical_tests.ipynb`
- ARIMA/VAR parameter significance tables.
- Diebold-Mariano tests (hybrid vs base and model vs baseline).
- Residual diagnostics (Ljung-Box, ARCH-LM, Jarque-Bera).
- Exports CSV summaries to `results/<target>/evaluation/`.

2. `khanh_model_analysis/02_robustness_checks.ipynb`
- Ramsey RESET specification-bias checks.
- Validation vs test error-distribution stability checks.
- Tail-event exclusion sensitivity for DM conclusions.

3. `khanh_model_analysis/03_results_interpretation.ipynb`
- Publication-ready ranking tables.
- DM evidence summary tables.
- Auto-generated interpretation draft with metrics and p-value evidence.

5. `khanh_model_analysis/05_forecast_visualization_and_economic_context.ipynb` (optional)
- Interactive top-model forecast dashboards and cumulative error curves.
- Event-overlay template for macro-financial annotation.

## Helper Module

Reusable statistical functions were added to:

- `src/statistical_validation.py`

## How To Run

From project root:

1. Run the pipeline first (if outputs are not ready):

```powershell
.\.venv\Scripts\python.exe main.py
```

2. Open and run notebooks in order:
- `khanh_model_analysis/00_comprehensive_time_series_analysis.ipynb`
- `khanh_model_analysis/01_statistical_tests.ipynb`
- `khanh_model_analysis/02_robustness_checks.ipynb`
- `khanh_model_analysis/03_results_interpretation.ipynb`
- `khanh_model_analysis/05_forecast_visualization_and_economic_context.ipynb` (optional)

3. Collect exported statistical tables from:

- `results/<active_target>/evaluation/`
- `results/<active_target>/diagnostics/`

## Notes on Extensibility

The notebooks do not hard-code pair names or fixed model lists. They discover available models from `results/<active_target>/` and load split data/paths via config, so adding ARIMAX/VARX/new hybrid variants requires only:

- updating config and/or pipeline outputs,
- then rerunning the notebooks.
