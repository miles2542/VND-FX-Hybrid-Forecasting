# Econometric Time-Series Explanation for Project Workflow (PHP)

## 1) Why this project is not just model comparison

This project is a full time-series econometric workflow, not only a leaderboard exercise. The core logic is:

1. Diagnose data-generating process properties (stationarity, breaks, dependence, heteroskedasticity).
2. Choose model architecture consistent with those properties.
3. Validate statistically (parameter significance, DM, residual diagnostics, robustness checks).
4. Interpret economically and identify stress windows where model risk is highest.

So the modeling decision follows data diagnostics, not arbitrary model preference.

## 2) Task-by-task mapping to your required workload

### 2.1 Tests

- Robustness checks:
	- RESET test for specification bias.
	- Validation-vs-test distribution stability (Mann-Whitney on absolute errors).
	- Tail-event exclusion sensitivity for DM conclusions.
- Parameter significance:
	- ARIMA parameter significance (z-stats, p-values).
	- VAR parameter significance (t-stats, p-values).
- DM and model-comparison significance:
	- DM tests across model pairs for MSE and MAE criteria.
	- Dedicated checks for hybrid vs linear core and hybrid vs baselines.

### 2.2 Results and interpretation structure

- CSV outputs are converted into interpretation-ready tables in notebooks.
- Ranking, DM, diagnostics, and abnormal-case scans are integrated in interpretation notebooks.
- Exported tables in `results/PHP/evaluation/reports/` now provide audit-ready evidence for thesis writing.

## 3) Why ARIMA/VAR first, then hybrid residual learners

The rationale is econometric, not cosmetic.

### 3.1 Linear memory justification

If Ljung-Box indicates serial dependence in returns or residuals, dynamic linear structures are needed:

$$
\mathbb{E}[r_t \mid r_{t-1}, r_{t-2}, ...] \neq 0
$$

This supports ARIMA/VAR as first-layer models.

### 3.2 Nonlinear and variance dynamics justification

If JB and ARCH-LM are significant, residuals are non-Gaussian and heteroskedastic:

$$
\mathrm{Var}(e_t \mid \mathcal{F}_{t-1}) = \sigma_t^2 \neq \sigma^2
$$

Then linear models are incomplete, so a nonlinear correction on residuals is justified:

$$
\hat{r}^{(H)}_t = \hat{r}^{(L)}_t + f_\psi\left(e^{(L)}_{t-1}, z_{t-1}\right)
$$

where $f_\psi$ is SVR/MLP.

### 3.3 Validation of this logic

The architecture is accepted only if metrics and DM tests support it out-of-sample. This is why DM and robustness checks are central, not optional.

## 4) What the current PHP outputs say (conceptual summary)

From the generated replication-audit table:

- RESET pass (non-constant models): 14.29% -> many models still face specification pressure.
- Val-test shift share: 80.00% -> sample instability is substantial.
- Tail-flip share: 37.27% -> conclusions are partly stress-event dependent.
- ARIMA significant param share: 66.67% -> linear signal exists.
- VAR significant param share: 31.43% -> spillovers are selective.
- DM MSE significant share: 60.91% -> model choice is often statistically material.
- LB/ARCH/JB rejection shares: 95.56% / 86.67% / 100.00% -> strong evidence for dependence, heteroskedasticity, and fat tails.

Interpretation: the results are exactly the type of mixed linear-plus-nonlinear structure where hybrid residual-learning models are theoretically and empirically appropriate.

## 5) Critical and problematic timeline windows (for time-series narrative)

Time-series projects must specify when the process is hardest to model, not only average performance.

Using `critical_windows_PHP.csv`, key stress windows include:

- CNYPHP: 2025-11-17 to 2025-11-27 (high stress, highest mean top errors).
- HKDPHP: 2026-03-19 to 2026-04-02.
- USDPHP: 2026-03-19 to 2026-04-02 (synchronized with HKD).
- JPYPHP: 2013-09-11 to 2013-09-24 (highest top-volatility mean among pairs).
- SGDPHP: 2012-07-10 to 2012-07-17 (highest ARCH stress share).

These are the windows where interpretation should focus on:

1. volatility regime shifts,
2. forecast-error clustering,
3. possible macro-policy/external shock alignment.

## 6) How to use these insights with external literature/news

Inside this code-only environment, we cannot ingest full external newspaper/paper text automatically. So the workflow should be:

1. Use the critical windows table to pick candidate event periods.
2. Match those windows with external macro-financial events manually.
3. Cite supporting papers/news to explain why specific windows were unstable.

[please input a paper to this insights to dive deeper]

Suggested insertion points for that external paper:

- After the critical windows list (to explain event-specific mechanisms).
- In the model-rationale section (to support hybrid architecture under EM volatility).
- In policy implications (to justify monitoring variables and reaction framework).

## 7) What is completed vs what can still improve

Completed:

- Baseline-paper replication tests are implemented and audited.
- Statistical significance and robustness evidence are exported and interpreted.
- Timeline-stress diagnostics are now explicit in notebook outputs and report narrative.

High-impact next steps:

1. Add ARIMAX/VARX with remittance and rate-differential proxies.
2. Add explicit EGARCH/GJR leverage-term estimation table for asymmetry claims.
3. Add rolling or regime-switching parameter models for non-stationary windows.
4. Add pair-level policy risk tiers based on stress-share and tail-flip metrics.
