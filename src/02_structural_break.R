# 02_structural_break.R - Bai-Perron Structural Break Detection
# Author: [Your Name]
# Purpose: Detect multiple structural breaks in stationary log-returns using Bai-Perron (2003) methodology.
# Citation: Bai, J., & Perron, P. (2003). "Computation and analysis of multiple structural change models." J. Applied Econometrics.
# Defense: We run Bai-Perron only on stationary log-returns (I(0)), not raw prices, and use HAC standard errors to account for volatility clustering (ARCH effects).

# Set up R environment
.libPaths(c("R_libs", .libPaths()))
library(jsonlite)
library(readr)
library(yaml)
library(strucchange)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
	stop("Config path must be provided as an argument.")
}
config_path <- args[1]
config <- read_yaml(config_path)

# Load processed data (must be log-returns, not raw prices)
processed_path <- file.path(config$paths$processed, "fx_aligned.csv")
data <- read_csv(processed_path, show_col_types = FALSE)

# Detect log-return columns
ret_cols <- grep("_RET$", names(data), value = TRUE)

results <- list()

# Runtime controls aligned with src/02_diagnostics.R
bp_h <- if (!is.null(config$diagnostics$structural_breaks$h)) config$diagnostics$structural_breaks$h else 0.15
bp_max_breaks <- if (!is.null(config$diagnostics$structural_breaks$max_breaks)) config$diagnostics$structural_breaks$max_breaks else 3
bp_compute_heavy_stats <- if (!is.null(config$diagnostics$structural_breaks$compute_heavy_stats)) config$diagnostics$structural_breaks$compute_heavy_stats else FALSE

cat("--- Running Bai-Perron Structural Break Detection (log-returns only) ---\n")

for (col in ret_cols) {
	cat(sprintf("--- Processing %s ---\n", col))
	tryCatch({
		valid_idx <- which(!is.na(data[[col]]))
		series <- data[[col]][valid_idx]
		series_dates <- data$Date[valid_idx]

		# Bai-Perron on log-returns with config-driven runtime controls.
		# Candidate models are searched up to bp_max_breaks and selected by BIC.
		ts_series <- ts(series)
		bp_candidates <- breakpoints(ts_series ~ 1, h = bp_h, breaks = bp_max_breaks)
		bic_vals <- BIC(bp_candidates)
		best_m <- which.min(bic_vals) - 1
		bp_model <- breakpoints(bp_candidates, breaks = best_m)

		heavy <- list(confint = list(), fstats = list())
		if (bp_compute_heavy_stats && !all(is.na(bp_model$breakpoints))) {
			confint_bp <- confint(bp_model, vcov = kernHAC)
			fstats <- Fstats(ts_series ~ 1, vcov = kernHAC)
			heavy <- list(
				confint = as.data.frame(confint_bp),
				fstats = as.data.frame(fstats$statistic)
			)
		}

		b_pts <- bp_model$breakpoints
		if (all(is.na(b_pts))) {
			breaks_found <- FALSE
			break_dates <- list()
		} else {
			breaks_found <- TRUE
			break_dates <- as.character(series_dates[b_pts])
		}

		results[[col]] <- list(
			breaks_found = breaks_found,
			break_indices = b_pts,
			break_dates = break_dates,
			n_breaks = ifelse(breaks_found, length(b_pts), 0),
			bic = as.numeric(bic_vals),
			confint = heavy$confint,
			fstats = heavy$fstats,
			method = sprintf("Bai-Perron (HAC option, h=%.2f, max_breaks=%d)", bp_h, bp_max_breaks)
		)

		cat(sprintf("[%s] Breaks found: %s | Dates: %s\n", col, breaks_found, paste(break_dates, collapse=", ")))
	}, error = function(e) {
		cat(sprintf("[ERROR] %s: %s\n", col, e$message))
		results[[col]] <<- list(error = e$message)
	})
}

# Save report
out_dir <- file.path(config$paths$results, "diagnostics")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
write_json(results, file.path(out_dir, "structural_breaks_report.json"), pretty = TRUE, auto_unbox = TRUE)

cat("--- Bai-Perron Structural Break Detection Completed. Report saved to results/diagnostics/structural_breaks_report.json ---\n")
