## Paper ↔ Project Divergences (Academic Context)

These are the critical methodological differences between the Ince & Trafalis (2006) study and this baseline template. Any team member or AI agent working on this project must adhere to these deviations to ensure academic rigor.

| Aspect | Paper (2006) | Our Project (2026) | Rationale |
|---|---|---|---|
| **Currencies** | EUR, GBP, JPY, AUD vs USD | USD, CNY, JPY, EUR vs **VND** | Transition to a managed-float VND-centric basket. |
| **Timeline** | 2000 – 2004 (~1700 obs) | 2010 – 2026 (~4200 trading days) | Ensures high-quality daily VND liquidity data from Yahoo Finance. |
| **Data sourcing** | Direct pairs | USD/VND native; others via cross-rates | Mitigates stale prices in direct VND pairs by using mathematically exact crosses. |
| **Splitting** | Random 3-way split | **Chronological** 80/10/10 split | Random splitting violates temporal ordering and leaks future info. *(Tashman, 2000)* |
| **Data Transform** | Raw price levels | **$100 \times$ log-returns** | Stabilizes optimization for ML solvers and provides interpretable % metrics. |
| **Hybrid logic** | ARIMA for lag selection only | **Residual hybrid** (Zhang 2003) | The Zhang (2003) approach is the standard for true linear-nonlinear hybrid forecasting. |
| **Scaling** | Not specified | Min-Max to [-1, 1] for ML | Required for SVR and MLP convergence; fit strictly on training set. |
| **CV for tuning** | 10-fold cross-validation | **TimeSeriesSplit** | Standard 10-fold is invalid for time series due to temporal dependency. |
| **Significance** | Paired t-test | **Diebold-Mariano Test** | DM tests are the standard for statistically comparing forecast accuracy. |
| **Structural breaks** | Ignored | Bai-Perron Test Hook | Prevents modeling across regimes with different data-generating processes. |
| **Stationarity** | ADF Only | ADF + Optional KPSS Hook | Provides a more robust unit-root diagnosis via stationarity-tests duality. |
