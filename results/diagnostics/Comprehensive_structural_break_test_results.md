## **Running the experiments**

- $h = 0.15, 0.10, 0.05$
- Type: mean, variance

```pwsh
🍫 ❯ uv run python run_bp_experiments.py
Bai-Perron Structural Break Orchestrator
Concurrency: 3 parallel runs, each using 4 threads.
Total target threads: 12 / 15 allowed.
--------------------------------------------------
FINISH h=0.15, mean in 324.89s
FINISH h=0.15, variance in 305.13s
FINISH h=0.1, mean in 635.56s
FINISH h=0.05, mean in 1104.63s
FINISH h=0.1, variance in 588.19s
FINISH h=0.05, variance in 823.64s
  Experiments ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:24:19
                    Bai-Perron Experiment Results                     
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Configuration    ┃ Status  ┃ Duration ┃ Result File                ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ h=0.05, mean     │ Success │ 1104.63s │ report_h0.05_mean.json     │
│ h=0.05, variance │ Success │  823.64s │ report_h0.05_variance.json │
│ h=0.1, mean      │ Success │  635.56s │ report_h0.10_mean.json     │
│ h=0.1, variance  │ Success │  588.19s │ report_h0.10_variance.json │
│ h=0.15, mean     │ Success │  324.89s │ report_h0.15_mean.json     │
│ h=0.15, variance │ Success │  305.13s │ report_h0.15_variance.json │
└──────────────────┴─────────┴──────────┴────────────────────────────┘
```

## **Results**

```pwsh
🍫 ❯ uv run python analyze_bp_results.py
╭────────────────────────────────────────────────╮
│ Bai-Perron Structural Break Diagnostic Summary │
╰────────────────────────────────────────────────╯
           Performance Metrics by Configuration            
 Config (h, type)  Avg Duration (s)  Total Config Time (s) 
 h=0.15, mean                319.55                1278.20
 h=0.1, mean                 627.61                2510.45
 h=0.05, mean               1099.66                4398.65
 h=0.15, variance            297.04                1188.16
 h=0.1, variance             581.38                2325.54
 h=0.05, variance            817.99                3271.97
--------------------------------------------------
                     Structural Breaks Discovery Matrix                      
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃         ┃ h=0.15 ┃ h=0.1  ┃ h=0.05 ┃   h=0.15   ┃   h=0.1    ┃   h=0.05   ┃
┃ Series  ┃ (mean) ┃ (mean) ┃ (mean) ┃ (variance) ┃ (variance) ┃ (variance) ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ USD/VND │   0    │   0    │   0    │     0      │     0      │     0      │
│ EUR/VND │   0    │   0    │   0    │     0      │     0      │     0      │
│ JPY/VND │   0    │   0    │   0    │     0      │     0      │     0      │
│ CNY/VND │   0    │   0    │   0    │     0      │     0      │     0      │
└─────────┴────────┴────────┴────────┴────────────┴────────────┴────────────┘

Note: No structural breaks were detected across all 6 configurations using the BIC selection criterion.
This suggests a high degree of regime stability in the log-returns (mean) and squared returns (variance) for these VND-centric pairs, or that the BIC penalty favored the zero-break model for the given sample size.
```
