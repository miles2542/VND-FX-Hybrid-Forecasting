# Dynamics and Volatility Forecasting of the PHP Exchange Rate: A Multi-Model Econometric Synthesis

## 1. Executive Summary (The Macro View)

This report integrates the executed notebook pipeline for the PHP target and interprets the results through an emerging-market sovereign risk lens. The evidence supports four high-confidence conclusions.

First, the PHP return system is stationary in levels for the analyzed pairs under standard unit-root frameworks (ADF and KPSS), so the short-run dynamics are statistically compatible with ARMA/VAR-class modeling without mandatory differencing in this setup. Second, the system exhibits strong contemporaneous co-movement and statistically meaningful cross-market transmission, especially through SGD and CNY channels, consistent with regional trade-finance linkage effects. Third, forecast residuals remain heteroskedastic, non-Gaussian, and often autocorrelated across models, indicating that simple linear structures are incomplete representations of risk dynamics. Fourth, hybrid residual-learning models (especially ARIMA+SVR and ARIMA+MLP) deliver the most reliable test-period error performance across most currency pairs, but robustness tests reveal stress-period instability in HKD, JPY, SGD, and USD links.

From a sovereign-risk perspective, the PHP process shows both mean-reverting short-horizon impulse behavior and intermittent instability pockets. Structural break tests do not strongly reject stable means at the **p < 0.05** threshold in sup-Chow outputs, but break-date concentrations around 2013, 2015, 2020, and 2021 coincide with plausible regime transitions (global liquidity cycles, pandemic disruptions, and post-pandemic inflation normalization).

Numerically, the model-risk profile is explicit:

- DM-MSE significance share is 60.9% (67/110 comparisons), meaning model choice is often economically material.
- DM-MAE significance share is 48.2% (53/110), showing robustness is weaker for absolute-error criteria than squared-error criteria.
- Tail-trimmed DM conclusions flip in 37.3% of comparisons (41/110), concentrated in HKD and USD pairs (15 flips each), indicating stress-regime fragility.
- Validation-to-test distribution shift is flagged in 80.0% of model-pair combinations (36/45), confirming substantial out-of-sample instability.

## 2. Rigorous Data Diagnostics

### 2.1 Stationarity and Integration Order

The stationarity panel indicates:

- ADF rejects a unit root for all five pairs (constant and constant+trend specifications), with p-values effectively near zero in many cases.
- KPSS does not reject stationarity (reported floor p-value = 0.10), implying no evidence against stationarity at conventional levels.
- Phillips-Perron fields are not populated in the exported panel, so PP-specific confirmation is unavailable in the current artifact set.

Additional numeric detail:

- ADF statistics range from about -11.49 to -51.17 across series/specifications, all far beyond conventional critical values.
- KPSS statistics are low (roughly 0.038 to 0.159), all with non-rejection at 5%.

Implication for integration order: the observed process is consistent with $I(0)$ returns in this project configuration. This validates a modeling strategy based on level return dynamics (ARIMA(0,0,1)-type residual structures, VAR in returns, and hybrid residual mapping), rather than differencing an already stationary process.

### 2.2 Cointegration and System Dependence

Johansen trace statistics reject at 95% for ranks $r=0$ through $r=4$, indicating a highly connected multivariate system. The rank-0 trace statistic is 7851.01 versus a 95% critical value of 69.82, so rejection is not marginal but overwhelming in magnitude. While return-stationarity and full-rank behavior can coexist in finite samples, the practical interpretation is clear: pair dynamics are not isolated, and multivariate spillover structure is strong.

The directional dependence matrix supports this:

- 14 of 20 directed Granger links are significant at 5% (70.0%).
- CNY and JPY each show 4/4 significant outward links, making them dominant transmitters in this system.
- USD and HKD each show 2/4 significant outward links, consistent with partial but not universal predictive spillovers.

Rolling-correlation evidence indicates persistent co-movement, not isolated episodes:

- USDPHP-HKDPHP mean rolling correlation is 0.9962 (std 0.0034; min 0.9844), effectively near-locked co-movement.
- USDPHP-CNYPHP mean rolling correlation is 0.8311.
- USDPHP-SGDPHP mean rolling correlation is 0.7325.
- JPY-linked pairs are materially lower and more variable (around 0.54-0.64 means with larger dispersion).

### 2.3 Structural Break Evidence (Chow/QLR Context)

The exported break table reports sup-Chow-type candidates but no significant break at **p < 0.05**:

| Pair | Candidate Date | p-value | 5% Break Decision |
|---|---:|---:|---|
| USDPHP_RET | 2013-01-22 | 0.1032 | No rejection |
| CNYPHP_RET | 2020-08-18 | 0.3931 | No rejection |
| JPYPHP_RET | 2015-06-03 | 0.2763 | No rejection |
| HKDPHP_RET | 2013-01-22 | 0.1051 | No rejection |
| SGDPHP_RET | 2021-12-14 | 0.2177 | No rejection |

No separate QLR test output is present in the saved artifacts; therefore, any QLR claim would be speculative. The conservative conclusion is: no formal 5% break rejection in the available sup-Chow panel, but date clustering still argues for rolling governance.

For macro interpretation, this means the first moment is not showing abrupt statistically verified jumps in this test, while second-moment instability and forecasting drift remain substantial in robustness diagnostics.

## 3. Model Architecture and Robustness

### 3.1 Why the Hybrid Architecture Fits PHP

The hybrid strategy decomposes the forecasting problem into a linear core plus nonlinear correction:

$$
y_t = \hat{y}^{(L)}_t + \hat{g}(e^{(L)}_{t-1}, x_{t-1}), \quad e^{(L)}_t = y_t - \hat{y}^{(L)}_t
$$

where $\hat{y}^{(L)}_t$ is a linear model forecast (ARIMA or VAR backbone), and $\hat{g}(\cdot)$ is a nonlinear learner (MLP or SVR) trained on residual structure. For an EM currency like PHP, this is economically coherent:

- linear channels capture policy-rate differential effects and broad external-balance drift,
- nonlinear residual channels absorb episodic shocks (risk-off events, liquidity squeezes, intervention windows),
- the combined form mitigates underfitting in crisis tails while preserving interpretability of the linear scaffold.

Cross-model averages quantify the architecture gain:

| Model | Avg Test MSE (100x) | Avg Test MAE (%) | Avg Test RMSE (%) |
|---|---:|---:|---:|
| hybrid_arima_svr | 0.3407 | 0.4134 | 0.5767 |
| hybrid_arima_mlp | 0.3446 | 0.4146 | 0.5803 |
| arima | 0.3801 | 0.4250 | 0.6116 |
| var | 0.3832 | 0.4308 | 0.6143 |
| baseline_rw | 0.4328 | 0.4402 | 0.6541 |

Thus, relative to baseline_rw, the best hybrid lowers average MSE by about $(0.4328-0.3407)/0.4328 \approx 21.3\%$.

### 3.2 Volatility Dynamics and Leverage-Effect Discussion

Residual diagnostics strongly support conditional heteroskedasticity and leptokurtic errors:

- ARCH-LM rejection share is 86.7% (39/45 model-pair tests),
- Jarque-Bera rejection share is 100.0% (45/45),
- Ljung-Box rejection share is 95.6% (43/45).

Model-level ARCH nuance is also informative:

- hybrid_var_svr rejects ARCH in 2/5 pairs (40%),
- hybrid_arima_svr rejects ARCH in 3/5 pairs (60%),
- most linear and baseline models reject in 5/5 pairs (100%).

This is consistent with volatility clustering and heavy tails in PHP-linked returns. The notebooks include GARCH-style residual diagnostics and persistence analysis, but no exported EGARCH/GJR asymmetry coefficient ($\gamma$) table is available in the current artifacts. Therefore, leverage asymmetry (bad-news vs good-news volatility reaction) cannot be formally confirmed from the saved outputs alone. The prudent statement is that clustering and fat tails are confirmed, while asymmetric sign-response remains a testable extension.

### 3.3 Robustness under Alternative Specifications

Three robustness layers were applied:

- RESET functional-form tests: only 5 of 35 non-constant fits pass (14.3%). hybrid_arima_svr passes in 4/5 pairs, the highest pass frequency.
- Validation-to-test stability (Mann-Whitney on absolute errors): 36 of 45 combinations show shift (80.0%). CNY is broadly stable; HKD, JPY, SGD, and USD show broad upward test MAE drift.
- Tail-sensitive DM re-tests: 41 of 110 comparisons flip significance status after trimming (37.3%), with flips concentrated in HKD and USD pairs.

Overall robustness verdict: ranking leadership of hybrid ARIMA residual learners remains broadly intact, but confidence in some pairwise significance claims should be stress-conditioned.

## 4. Results Interpretation and Forecasting Performance

### 4.1 Error Metrics and Economic Meaning

Best test MSE by pair (100x scale metric from exported summary):

| Pair | Best Model | Test MSE (100x) | Test RMSE (%) |
|---|---|---:|---:|
| USDPHP_RET | hybrid_arima_svr | 0.2812 | 0.5303 |
| CNYPHP_RET | hybrid_arima_mlp | 0.2962 | 0.5443 |
| JPYPHP_RET | hybrid_arima_svr | 0.5708 | 0.7555 |
| HKDPHP_RET | hybrid_arima_svr | 0.2804 | 0.5296 |
| SGDPHP_RET | hybrid_arima_svr | 0.2657 | 0.5155 |

Relative gains versus random walk baseline are economically large in four of five pairs:

| Pair | Best Model | MSE Improvement vs baseline_rw |
|---|---|---:|
| USDPHP_RET | hybrid_arima_svr | 27.34% |
| CNYPHP_RET | hybrid_arima_mlp | 22.73% |
| JPYPHP_RET | hybrid_arima_svr | 9.52% |
| HKDPHP_RET | hybrid_arima_svr | 27.36% |
| SGDPHP_RET | hybrid_arima_svr | 29.43% |

Interpretation in business units: if spot USDPHP is near 56.0, a 0.53% RMSE implies an average absolute error scale of roughly $56.0 \times 0.0053 \approx 0.30$ PHP, i.e., about 30 centavos. For importers with thin pass-through margins, a 20-40 centavo forecast band can materially affect pricing, hedging tenor choice, and inventory valuation. At large monthly invoice sizes, this forecast-error magnitude can dominate operating margin fluctuations.

### 4.2 DM Evidence and Economic Materiality

DM tests show frequent significant outperformance of ARIMA/hybrid families versus naive baselines at **p < 0.05**. Quantitatively:

- MSE-based DM significance: 60.9% of all tested pairwise comparisons.
- MAE-based DM significance: 48.2% of all tested comparisons.
- This MSE vs MAE gap indicates that relative model advantages are stronger in large-error episodes than in median-like days.

Head-to-head differences among strong models are still pair-dependent and often near-threshold (for example several p-values in the 0.05-0.10 band), implying model diversification across top-tier candidates remains rational for treasury operations.

### 4.3 Visual Forecast Behavior (Trend and Uncertainty)

The forecast-visualization notebook exports trend-comparison and error-path charts. While a formal probabilistic fan chart with explicit quantile bands is not exported as a table, model-spread and error clustering in test windows imply widening uncertainty during stress subperiods, especially for JPY-linked and post-shift regimes. In practice, this is equivalent to a state-dependent confidence envelope that should be broader when volatility clusters intensify.

The ranking evidence supports this visual interpretation: JPY has the highest best-model MSE (0.5708), more than double SGDPHP's best-model MSE (0.2657). This spread is consistent with a wider effective forecast cone in JPY-linked exposures.

## 5. Economic Context and Policy Implications

### 5.1 Remittance Seasonality and PHP Dynamics

The current model outputs do not include an explicit OFW remittance seasonal regressor or a quarter-specific decomposition table. Therefore, direct statistical attribution of Q4 appreciation to remittance inflows is not established by the saved diagnostics. Still, the framework can accommodate this by adding seasonal dummies/interactions or external remittance-flow covariates in ARIMAX/VARX extensions. Given known Philippine balance-of-payments seasonality, this is a high-priority upgrade.

An implementable specification is:

$$
r_t = \alpha + \sum_{i=1}^{p}\phi_i r_{t-i} + \beta_1 \text{Remit}_t + \beta_2 D_{Q4,t} + \beta_3(\text{Remit}_t \times D_{Q4,t}) + u_t
$$

with corresponding volatility block $u_t=\sigma_t z_t$ and GARCH/EGARCH dynamics on $\sigma_t^2$.

### 5.2 BSP-Relevant Monitoring Signals

Given confirmed heteroskedasticity, heavy tails, and split instability, policy and surveillance focus should include:

- high-frequency volatility clustering metrics and ARCH persistence,
- stress-window forecast error drift between validation/test analogues,
- spillover-sensitive pairs (USD, SGD, CNY channels) for early warning,
- intervention effectiveness diagnostics that compare post-intervention residual variance decay.

For macroprudential implementation, the evidence supports adaptive reserve-liquidity management and dynamic hedging guidance rather than fixed-rule assumptions of stable variance.

Numerically, two triggers are especially policy-relevant:

- when validation-to-test MAE drift exceeds about 8-12 bps (as seen repeatedly outside CNY),
- when DM significance flips under tail trimming for key policy pairs (USD/HKD channels), signaling unstable model ranking under stress.

### 5.3 Mean Reversion versus Persistence

Impulse response functions show fast own-shock decay toward zero by horizons roughly 6-10 steps, consistent with short-run mean reversion in return impulses. Simultaneously, residual volatility persistence and recurrent ARCH significance imply persistent second-moment risk. Thus, the PHP appears mean-reverting in conditional mean innovations but persistence-prone in conditional variance.

For example, the USDPHP own IRF moves from 1.00 at impact to approximately -0.0286 at step 1 and near zero by later horizons, while cross-shock responses (for example from SGD and CNY) are economically nontrivial at step 1 before decaying. This is the empirical signature of transitory mean effects with more persistent volatility risk.

## 6. Technical Appendix (Model Equations)

### 6.1 ARIMA Backbone

$$
\phi(B)(y_t - \mu) = \theta(B)\varepsilon_t, \quad \varepsilon_t \sim (0,\sigma_t^2)
$$

For ARIMA$(0,0,1)$:

$$
y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1}
$$

### 6.2 VAR(1) Backbone

$$
\mathbf{y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{y}_{t-1} + \mathbf{u}_t
$$

### 6.3 Hybrid Residual Learning

$$
\hat{y}^{(H)}_t = \hat{y}^{(L)}_t + f_\psi\!\left(\hat{e}^{(L)}_{t-1}, \mathbf{z}_{t-1}\right)
$$

with $f_\psi$ implemented as SVR or MLP on residual features.

### 6.4 GARCH(1,1)

$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2,
\quad \omega>0,\; \alpha,\beta\ge 0
$$

Persistence measure: $\alpha+\beta$.

### 6.5 EGARCH (for Future Asymmetry Testing)

$$
\log(\sigma_t^2)=\omega+\beta\log(\sigma_{t-1}^2)+\alpha\left(\frac{|\varepsilon_{t-1}|}{\sigma_{t-1}}-\mathbb{E}\left[\frac{|\varepsilon_{t-1}|}{\sigma_{t-1}}\right]\right)+\gamma\frac{\varepsilon_{t-1}}{\sigma_{t-1}}
$$

Leverage effect test: $\gamma<0$ indicates stronger volatility response to negative shocks.

## 7. Replication Task Coverage (Against Baseline-Paper Workflow)

This section maps the implemented pipeline to your requested replication tasks.

### 7.1 Tests Coverage Matrix

From the generated audit table (`results/PHP/evaluation/reports/replication_audit_PHP.csv`):

| Requested Task | Metric from Outputs | Value | Interpretation |
|---|---|---:|---|
| Robustness/specification bias | RESET pass share (non-constant) | 14.29% | Most models still show misspecification risk; nonlinear augmentation remains justified. |
| Robustness/sample stability | Val-vs-test shift share | 80.00% | Strong evidence of regime/sample instability out-of-sample. |
| Robustness/tail dependence | DM tail-flip share | 37.27% | Over one-third of pairwise conclusions are tail-sensitive. |
| Param significance (ARIMA) | Significant parameter share | 66.67% | Linear ARIMA components are materially informative. |
| Param significance (VAR) | Significant parameter share | 31.43% | Spillovers exist but are selective, not dense in every equation. |
| DM overall | DM MSE significant share | 60.91% | Model choice frequently changes accuracy materially. |
| DM hybrid vs linear base | Hybrid>Ari/Var significant share | 20.00% | Hybrids dominate baselines strongly, but hybrid-vs-strong-linear is pair-dependent. |
| Diagnostics (LB) | Rejection share | 95.56% | Residual serial dependence remains widespread. |
| Diagnostics (ARCH-LM) | Rejection share | 86.67% | Conditional variance dynamics are persistent. |
| Diagnostics (JB) | Rejection share | 100.00% | Leptokurtic/non-Gaussian residuals are universal in this run. |

Conclusion on task completion: all three required test blocks were implemented and evidenced in exported outputs: robustness checks, parameter significance tests, and DM-based statistical model comparison.

### 7.2 Why Hybrid Is Rational (Econometric Logic Chain)

The replication logic is coherent and empirically supported:

1. LB significance on returns/residuals indicates linear dynamic memory, supporting ARIMA/VAR as layer 1.
2. JB and ARCH-LM significance indicate non-normal, heteroskedastic residual structure that linear models do not fully capture.
3. Residual-learning SVR/MLP layer targets remaining nonlinear error structure.
4. Metrics and DM tests then verify whether that architecture improves realized forecast performance.

Formally, if layer-1 residuals satisfy:

$$
\mathbb{E}[e_t \mid \mathcal{F}_{t-1}] \neq 0 \quad \text{or} \quad \mathrm{Var}(e_t \mid \mathcal{F}_{t-1}) \neq \sigma^2,
$$

then an additional nonlinear correction stage is econometrically justified.

## 8. Critical Time Windows and Problematic Segments (PHP)

Using the generated table (`results/PHP/evaluation/reports/critical_windows_PHP.csv`), the project now explicitly flags where the time-series problem is most difficult:

| Pair | Top Volatility Window | ARCH Stress Share | Mean of Top Errors | Break Anchor |
|---|---|---:|---:|---|
| CNYPHP_RET | 2025-11-17 to 2025-11-27 | 70.04% | 3.8883 | 2020-08-18 |
| HKDPHP_RET | 2026-03-19 to 2026-04-02 | 61.02% | 3.7195 | 2013-01-22 |
| JPYPHP_RET | 2013-09-11 to 2013-09-24 | 56.65% | 3.3368 | 2015-06-03 |
| SGDPHP_RET | 2012-07-10 to 2012-07-17 | 76.28% | 3.4909 | 2021-12-14 |
| USDPHP_RET | 2026-03-19 to 2026-04-02 | 61.50% | 3.7277 | 2013-01-22 |

Interpretation:

- SGDPHP has the highest ARCH-stress share (76.28%), so variance-regime instability is strongest there.
- CNYPHP has the highest top-error mean (3.8883), suggesting largest episodic forecast misses.
- USD and HKD share the same top volatility window in late Mar-early Apr 2026, indicating synchronized stress transmission.
- Structural-break anchors and stress windows are not identical in timing, which is typical when mean and variance shifts occur through different channels.

This timeline framing is critical for your paper narrative because it tells where the model is most informative, where it is most vulnerable, and where external event matching (policy/news) should focus.

[please input a paper to this insights to dive deeper]

## 9. Final Strategic Assessment

The empirical evidence supports a clear strategic ranking: hybrid ARIMA residual-learning models are the most effective forecasting family for PHP-linked returns in this dataset, with hybrid_arima_svr the most consistently competitive performer across pairs. However, policy-grade deployment must account for heteroskedasticity, leptokurtic tails, and regime-sensitive robustness breakdowns.

For sovereign risk management and market strategy, the correct posture is not static model selection but adaptive model governance: keep hybrid leaders as primary engines, retain challengers for drift detection, and embed stress-conditional monitoring tied to BSP policy windows, external-rate shocks, and seasonal liquidity cycles.

Operationally, a three-tier deployment protocol is recommended:

1. Primary engine: hybrid_arima_svr (default), hybrid_arima_mlp (challenger).
2. Triggered review: activate when rolling error drift and tail-flip diagnostics breach thresholds.
3. Policy escalation: widen hedging bands and shorten forecast refresh cycles when variance-cluster alerts intensify.
