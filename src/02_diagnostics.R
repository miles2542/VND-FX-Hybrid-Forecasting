# 02_diagnostics.R - Skeleton for Stationarity and Structural Breaks
# Lead role: Implement paper's tests (ADF).
# Member role: Implement gaps (KPSS, Bai-Perron).

# Purely relative path (R handles this better than bracketed absolute paths)
.libPaths(c("R_libs", .libPaths()))

library(jsonlite)
library(tseries)
library(readr)
library(yaml)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Config path must be provided as an argument.")
}

config_path <- args[1]
config <- read_yaml(config_path)

processed_path <- file.path(config$paths$processed, "fx_aligned.csv")
data <- read_csv(processed_path, show_col_types = FALSE)

# Detect log-return columns
ret_cols <- grep("_RET$", names(data), value = TRUE)

results <- list()

cat("--- Running Diagnostics (Paper Baseline) ---\n")

for (col in ret_cols) {
  series <- na.omit(data[[col]])
  
  # 1. ADF Test (Implementing Paper Method)
  # ----------------------------------------
  # Paper: Page 6, "found to be I(1) using ADF"
  # Since we are using log-returns, they should be I(0)
  adf <- adf.test(series, alternative = "stationary")
  
  # 2. KPSS Test (Skeleton for Member D)
  # ----------------------------------------
  # HOOK: Member D should implement confirmatory stationarity analysis here
  kpss_p_value <- NA # Placeholder
  
  # 3. Structural Break Detection (Skeleton for Member E)
  # ----------------------------------------
  # HOOK: Member E should implement Bai-Perron (strucchange::breakpoints) here
  # Default: Assume no breaks for the baseline run
  breaks_found <- FALSE
  break_dates <- list()
  
  results[[col]] <- list(
    adf = list(
      statistic = as.numeric(adf$statistic),
      p_value = as.numeric(adf$p.value),
      conclusion = ifelse(adf$p.value < 0.05, "Stationary (Reject Unit Root)", "Non-Stationary")
    ),
    kpss = list(
      p_value = kpss_p_value,
      status = "PENDING - Member D Task"
    ),
    structural_breaks = list(
      found = breaks_found,
      dates = break_dates,
      status = "PENDING - Member E Task"
    )
  )
  
  cat(sprintf("[%s] ADF p-value: %.4f | Conclusion: %s\n", 
              col, adf$p.value, results[[col]]$adf$conclusion))
}

# Save report
out_dir <- file.path(config$paths$results, "diagnostics")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
write_json(results, file.path(out_dir, "initial_report.json"), pretty = TRUE, auto_unbox = TRUE)

cat("--- Diagnostics Completed. Report saved to results/diagnostics/initial_report.json ---\n")
