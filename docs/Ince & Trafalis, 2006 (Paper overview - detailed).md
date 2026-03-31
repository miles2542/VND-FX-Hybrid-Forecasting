# <a id='toc2_'></a>[**#3: Ince & Trafalis (2006) – "A hybrid model for exchange rate prediction"**](#toc0_)

### <a id='toc2_1_1_'></a>[**1. Meta & Credibility**](#toc0_)
*   **Journal**: *Decision Support Systems* (Elsevier). ABDC A* rating. Highly credible.
*   **Domain**: Financial Econometrics / Forex Prediction.
*   **Core Premise**: Two-stage hybrid forecasting. Use classical parametric time series models (ARIMA/VAR) for feature selection (determining optimal lags), then feed those lags into non-parametric machine learning models (SVR/MLP) for prediction. 

### <a id='toc2_1_2_'></a>[**2. Experimental Blueprint (Replication Data)**](#toc0_)
*Must be replicated exactly for the baseline.*
*   **Data Scope**: Daily exchange rates (Jan 1, 2000 – May 26, 2004). Total N ≈ 1704.
*   **Variables**: EUR/USD, GBP/USD, JPY/USD, AUD/USD.
*   **Data Split**: Train (1544 obs), 10-fold Cross-Validation (100 obs), Test (60 obs). 
*   **Stage 1: Input Selection (Econometrics)**
    *   **Stationarity**: ADF test confirmed all series non-stationary (I(1)).
    *   **Univariate Selection (ARIMA)**: Box-Jenkins methodology (ACF/PACF + AIC/BIC).
        *   Identified baselines: EUR(3,1,0), GBP(4,1,0), JPY(3,1,0), AUD(2,1,0).
        *   *Meaning*: EUR prediction requires 3 past lags; GBP requires 4, etc.
    *   **Multivariate Selection (VAR/Co-integration)**: 
        *   Johansen test: No co-integration found among the 4 currencies. 
        *   VAR Lag length: $k=4$ selected via Likelihood Ratio (LR) and AIC.
        *   Granger Causality: EUR & GBP mutually causal. JPY & AUD independent (rely only on their own 4 lags).
*   **Stage 2: Prediction (Machine Learning)**
    *   **Models**: MLP (1 hidden layer, Backpropagation) vs. SVR ($\epsilon$-insensitive, RBF kernel).
    *   **Inputs**: Lags determined in Stage 1. 
    *   **Metrics**: MSE, MAE, paired t-tests for significance.
*   **Results**: 
    *   Hybrid SVR outperforms Hybrid MLP.
    *   Hybrid models outperform pure ARIMA/VAR baseline (MSE comparison).
    *   ARIMA input selection works best for SVR; VAR input selection works best for MLP.

### <a id='toc2_1_3_'></a>[**3. Replication Feasibility (Technical Assessment)**](#toc0_)
*   **Difficulty**: Low/Moderate. Highly feasible for a 1-month group project.
*   **Data Acquisition**: Trivial. Yahoo Finance (`yfinance` API) or FRED.
*   **Tooling**: Python. `statsmodels` (ADF, ARIMA, VAR, Granger), `scikit-learn` (SVR, MLP, GridSearch for CV).
*   **Bottleneck Risk**: SVR hyperparameter tuning ($C$, $\gamma$, $\epsilon$). Authors used 10-fold CV. Easily handled via standard `GridSearchCV`.

### <a id='toc2_1_4_'></a>[**4. Academic Vulnerabilities & Weaknesses (The "Attack Surface")**](#toc0_)
*   **Ancient Data**: 2000-2004 timeline misses modern market dynamics (2008 GFC, Algorithmic Trading dominance, COVID-19 shock).
*   **Endogeneity Trap**: Purely autoregressive. Forecasts rely *only* on past exchange rates. Ignores macroeconomic fundamentals (interest rates, inflation differentials) which violate standard economic theory (e.g., Uncovered Interest Rate Parity).
*   **Volatility Ignored**: FX markets are notoriously heteroskedastic (volatility clustering). Authors mention GARCH in intro but fail to use it.
*   **Static Testing**: Test set is a static 60-day window. Weak robustness. FX regimes shift dynamically.

### <a id='toc2_1_5_'></a>[**5. Project Directions & Gap Exploitation**](#toc0_)
*Select one narrative to frame your thesis.*

**Option A: The Macroeconomic Fundamentalist (Highly Recommended)**
*   **The Gap**: Method/Context. Paper models FX in a vacuum.
*   **The Fix**: Upgrade Stage 1. Replace ARIMA with **ARIMAX** (or VAR with VARX). 
*   **Execution**: 
    1. Replicate exact 2006 baseline on modern data (e.g., 2015-2023).
    2. Introduce exogenous variables (Interest Rate differentials between US/EU, Inflation rates, or VIX index).
    3. Prove that ARIMAX-SVR outperforms pure ARIMA-SVR. 
*   **Defense**: Backed by standard macroeconomic theory. Demonstrates mastery of exogenous variable integration in time series.

**Option B: The Volatility / Risk Manager Approach**
*   **The Gap**: Method. Paper ignores variance clustering.
*   **The Fix**: Replace/augment ARIMA with **ARIMA-GARCH**.
*   **Execution**: 
    1. Replicate baseline. Check residuals for ARCH effects (Engle's test). 
    2. Build a Stage 1 model that captures both mean (ARIMA) and variance (GARCH). 
    3. Feed standardized residuals or volatility estimates into the SVR.
*   **Defense**: Hits core Time Series syllabus (GARCH is a staple). Shows deep understanding of financial time series properties (heteroskedasticity).

**Option C: The Structural Break Stress Test**
*   **The Gap**: Data/Context. Paper uses a calm, static market window.
*   **The Fix**: Regime evaluation. 
*   **Execution**: 
    1. Pull data from 2018–2026. 
    2. Implement baseline.
    3. Split evaluation into Pre-COVID (2018-2019), COVID crash (2020), and Inflation crisis (2022-2023), ... 
    4. Implement **Rolling Window Forecasting** instead of a static 60-day test set. Evaluate how the ARIMA-SVR hybrid degrades during structural breaks. 
*   **Defense**: Textbook application of structural break theory and robust backtesting. Easiest technically, highest analytical yield.

### <a id='toc2_1_6_'></a>[**6. Final Verdict & Strategy Notes**](#toc0_)
*   **Status**: Excellent candidate for the primary paper.
*   **Why**: It bridges traditional econometrics (satisfying the course requirement) with ML (allowing high grades for "advanced" application). The math is transparent. The gaps are glaring but easily fixable using standard syllabus tools (ARIMAX, GARCH, Rolling Windows). 
*   **Action for Miles**: If choosing this, assign one teammate immediately to build the data pipeline (`yfinance` pull for EUR, GBP, JPY, AUD vs USD, daily). Assign another to write the `statsmodels` ARIMA baseline.
