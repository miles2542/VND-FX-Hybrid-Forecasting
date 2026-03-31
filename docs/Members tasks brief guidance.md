## Team Member Extension Hooks (Contribution Roadmap)

These extensions are not part of the Lead's architectural baseline. Team members are responsible for implementing logic at these designated entry points to achieve the project's academic goals.

| Hook | Location | Description | Primary Owner |
|---|---|---|---|
| **ARIMAX / VARX** | `src/03_parametric.R` | Integrate lagged interest rate differentials (`IR_DIFF_LAGGED`) into models. | Member A |
| **Granger Causality** | `src/03_parametric.R` | Execute and interpret `vars::causality()` for the VND basket. | Member A |
| **VECM Integration** | `src/03_parametric.R` | If Johansen test finds co-integration, implement `urca::cajorls()` logic. | Member A |
| **Vietnam Rate Data** | `src/01_data_loader.py` | Replace zero-filled `vn_interest_rate_raw.csv` with validated SBV data. | Member B |
| **Denton-Chow-Lin** | `src/01_data_loader.py` | Add temporal disaggregation for monthly/quarterly macro variables (GDP, M2). | Member B |
| **Rolling Origin CV** | `main.py` wrapper | Wrap the orchestrator in a rolling cross-validation loop (Expanding/Sliding). | Member C |
| **Diebold-Mariano** | `src/05_evaluation.py` | Replace the `diebold_mariano_test` stub with actual S-statistic logic. | Member D |
| **Residual Diagnostics** | `src/05_evaluation.py` | Add ACF, Q-Q, and Breusch-Godfrey plots to the evaluation output. | Member D |
| **Structural Breaks** | `src/02_diagnostics.R` | Implement Bai-Perron detection to validate regime stability (2010–2026). | Member E |
| **KPSS Stationarity** | `src/02_diagnostics.R` | Implement KPSS test to cross-verify ADF stationarity findings. | Member E |
| **Thesis Tables** | `scripts/` (New) | Create automated LaTeX/Markdown table generators for the final paper. | All |

**Note for Developers:** Implementation should prioritize modularity. Ensure new functions do not break the "Polyglot Handoff" (CSV/JSON communication) between Python and R.
