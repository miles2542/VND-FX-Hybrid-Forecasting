# Data Time Range Selection: 2010–2026

### Selection
- **Start Date**: Jan 1, 2010
- **End Date**: Mar 31, 2026 (approx. 4,200 observations)

### Rationale: Quality vs. Quantity
While `USDVND=X` has data back to 2003, the 2010 cutoff was selected to prioritize econometric homogeneity:

1. **Exchange Rate Regime Transition**: Pre-2010 VND was heavily pegged, resulting in "step-function" data with low information content. Post-2010 data captures the shift toward the managed float (e.g., 2016 Central Exchange Rate mechanism), which is more representative for predictive modeling.
2. **Reduced Data Sparsity**: Early 2000s cross-rates (especially VND and CNY peers) exhibit frequent stale prices and unrecorded trading days on retail feeds (`yfinance`). The 2010 onwards data maintains high-frequency daily consistency.
3. **Macroeconomic Stability**: By starting in 2010, the model avoids overfitting to the extreme volatility of the 2008 Global Financial Crisis (GFC), focusing instead on more modern "stressed" yet standard market conditions (e.g., 2011 inflation spike, COVID-19).
4. **Sample Size Suitability**: The ~4,200-day window is significantly larger than the original Ince & Trafalis (2006) dataset (~1,700 days), providing ample observations for deep learning (MLP) and high-lag (p=10) VAR models.
