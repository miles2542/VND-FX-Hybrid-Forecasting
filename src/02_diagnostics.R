# 02_diagnostics.R - Skeleton for Stationarity and Structural Breaks
# Lead role: Implement paper's tests (ADF).
# Member role: Implement gaps (KPSS, Bai-Perron).

# Purely relative path (R handles this better than bracketed absolute paths)
.libPaths(c("R_libs", .libPaths()))

library(jsonlite)
library(urca) # For ur.df (ADF) and ur.kpss (KPSS) tests
library(readr)
library(yaml)
library(strucchange) # Library required for the Bai-Perron test
library(sandwich)    # For HAC standard errors (kernHAC)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Config path must be provided as an argument.")
}

config_path <- args[1]
config <- read_yaml(config_path)

processed_path <- file.path(config$paths$processed, config$active_target, "fx_aligned.csv")
data <- read_csv(processed_path, show_col_types = FALSE)

# Detect log-return columns
ret_cols <- grep("_RET$", names(data), value = TRUE)

results <- list()

# Section switches: run all diagnostics in one script, or selectively skip parts.
run_adf_kpss <- if (!is.null(config$diagnostics$run_adf_kpss)) config$diagnostics$run_adf_kpss else TRUE
run_structural_breaks <- if (!is.null(config$diagnostics$run_structural_breaks)) config$diagnostics$run_structural_breaks else TRUE

# Parallel execution controls (Windows-safe via parallel::parLapply)
parallel_enabled <- if (!is.null(config$diagnostics$parallel$enabled)) config$diagnostics$parallel$enabled else FALSE
max_retries_per_series <- if (!is.null(config$diagnostics$parallel$max_retries_per_series)) config$diagnostics$parallel$max_retries_per_series else 1
retry_backoff_seconds <- if (!is.null(config$diagnostics$parallel$retry_backoff_seconds)) config$diagnostics$parallel$retry_backoff_seconds else 1
fallback_to_sequential_on_parallel_error <- if (!is.null(config$diagnostics$parallel$fallback_to_sequential_on_parallel_error)) config$diagnostics$parallel$fallback_to_sequential_on_parallel_error else TRUE

# Worker selection policy:
# - If n_workers is omitted, auto-pick from available cores and number of series.
# - If n_workers is provided, sanitize and cap at number of series.
available_cores <- parallel::detectCores(logical = TRUE)
if (is.na(available_cores) || available_cores < 1) {
  available_cores <- 2
}
auto_workers <- max(1, min(length(ret_cols), max(1, available_cores - 1), 4))

if (!is.null(config$diagnostics$parallel$n_workers)) {
  parallel_n_workers_requested <- as.integer(config$diagnostics$parallel$n_workers)
  if (is.na(parallel_n_workers_requested) || parallel_n_workers_requested < 1) {
    parallel_n_workers_requested <- 1
  }
  parallel_worker_source <- "config"
} else {
  parallel_n_workers_requested <- auto_workers
  parallel_worker_source <- "auto"
}

parallel_n_workers <- min(parallel_n_workers_requested, length(ret_cols))

# Bai-Perron runtime controls (can be overridden from config if provided)
bp_h <- if (!is.null(config$diagnostics$structural_breaks$h)) config$diagnostics$structural_breaks$h else 0.15
bp_max_breaks <- if (!is.null(config$diagnostics$structural_breaks$max_breaks)) config$diagnostics$structural_breaks$max_breaks else 3
bp_compute_heavy_stats <- if (!is.null(config$diagnostics$structural_breaks$compute_heavy_stats)) config$diagnostics$structural_breaks$compute_heavy_stats else FALSE
bp_targets <- if (!is.null(config$diagnostics$structural_breaks$targets)) config$diagnostics$structural_breaks$targets else "mean"
bp_variance_proxy <- if (!is.null(config$diagnostics$structural_breaks$variance_proxy)) config$diagnostics$structural_breaks$variance_proxy else "squared"

# Prepare output path once and write run metadata immediately to avoid stale-file confusion.
out_dir <- file.path(config$paths$results, config$active_target, "diagnostics")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
report_path <- file.path(out_dir, "initial_report.json")

results[["_meta"]] <- list(
  run_started = as.character(Sys.time()),
  status = "running",
  run_adf_kpss = run_adf_kpss,
  run_structural_breaks = run_structural_breaks,
  parallel_enabled = parallel_enabled,
  parallel_worker_source = parallel_worker_source,
  parallel_n_workers_requested = parallel_n_workers_requested,
  parallel_n_workers = parallel_n_workers,
  available_cores = available_cores,
  max_retries_per_series = max_retries_per_series,
  retry_backoff_seconds = retry_backoff_seconds,
  fallback_to_sequential_on_parallel_error = fallback_to_sequential_on_parallel_error,
  bp_h = bp_h,
  bp_max_breaks = bp_max_breaks,
  bp_compute_heavy_stats = bp_compute_heavy_stats,
  bp_targets = bp_targets,
  bp_variance_proxy = bp_variance_proxy,
  config_path = config_path
)
write_json(results, report_path, pretty = TRUE, auto_unbox = TRUE)

cat("--- Running Diagnostics (Paper Baseline) ---\n")


# --- Stationarity Testing: ADF (ur.df) & KPSS (ur.kpss) ---
# ADF: Dickey, D.A. & Fuller, W.A. (1979). Distribution of the Estimators for Autoregressive Time Series With a Unit Root. JASA.
# KPSS: Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. J Econometrics.
# Dual testing is recommended for robust inference (see Schwert, 1989).


# Add safe parallel execution with retry and sequential fallback.
process_single_series <- function(col, show_structure = FALSE) {
  valid_idx <- which(!is.na(data[[col]]))
  if (length(valid_idx) == 0) {
    stop(sprintf("No non-NA observations for %s", col))
  }

  series <- data[[col]][valid_idx]
  series_dates <- data$Date[valid_idx]
  messages <- c()

  adf_block <- list(status = "skipped")
  kpss_block <- list(status = "skipped")
  structural_breaks_block <- list()
  if (!run_structural_breaks) {
    structural_breaks_block[["_status"]] <- "Skipped by config (diagnostics.run_structural_breaks=false)"
  }

  if (run_adf_kpss) {
    adf <- ur.df(series, type = "drift", selectlags = "AIC")
    adf_stat <- if ("tau2" %in% names(adf@teststat)) adf@teststat["tau2"] else adf@teststat[[1]]
    adf_cval <- if (!is.null(dimnames(adf@cval)[[1]]) && "tau2" %in% dimnames(adf@cval)[[1]]) adf@cval["tau2",] else adf@cval[1,]
    adf_decision <- if (!is.na(adf_stat) && !is.na(adf_cval["5pct"])) {
      ifelse(adf_stat < adf_cval["5pct"], "Stationary (Reject Unit Root)", "Non-Stationary")
    } else {
      NA
    }

    kpss <- ur.kpss(series, type = "mu", lags = "short")
    kpss_stat <- kpss@teststat
    kpss_cval <- kpss@cval
    if (is.null(names(kpss_cval))) {
      names(kpss_cval) <- c("10pct", "5pct", "2.5pct", "1pct")
    }
    kpss_decision <- if (!is.na(kpss_stat) && !is.na(kpss_cval["5pct"])) {
      ifelse(kpss_stat > kpss_cval["5pct"], "Non-Stationary (Reject Null)", "Stationary (Fail to Reject)")
    } else {
      NA
    }

    adf_block <- list(
      statistic = adf_stat,
      critical_values = adf_cval,
      decision = adf_decision
    )
    kpss_block <- list(
      statistic = kpss_stat,
      critical_values = kpss_cval,
      decision = kpss_decision
    )

    if (show_structure) {
      messages <- c(messages, sprintf("[%s] ADF teststat names: %s", col, paste(names(adf@teststat), collapse = ", ")))
      messages <- c(messages, sprintf("[%s] ADF cval rows: %s", col, paste(dimnames(adf@cval)[[1]], collapse = ", ")))
      messages <- c(messages, sprintf("[%s] KPSS cval names: %s", col, ifelse(is.null(names(kpss@cval)), "NULL", paste(names(kpss@cval), collapse = ", "))))
    }

    messages <- c(messages, sprintf("[%s] ADF stat: %.4f | 5%% cval: %.4f | Decision: %s", col, adf_stat, adf_cval["5pct"], adf_decision))
    messages <- c(messages, sprintf("[%s] KPSS stat: %.4f | 5%% cval: %.4f | Decision: %s", col, kpss_stat, kpss_cval["5pct"], kpss_decision))
  } else {
    messages <- c(messages, sprintf("[%s] ADF/KPSS skipped by config.", col))
  }

  if (run_structural_breaks) {
    for (target in bp_targets) {
      messages <- c(messages, sprintf("[%s] Running Bai-Perron for %s (h=%.2f, max_breaks=%d)...", col, target, bp_h, bp_max_breaks))
      
      # Prepare series based on target
      if (target == "variance") {
        # Proxy: (y - mean)^2 or |y - mean|
        demeaned <- series - mean(series)
        if (bp_variance_proxy == "squared") {
          ts_series <- ts(demeaned^2)
          method_label <- "Bai-Perron on Squared Residuals"
        } else if (bp_variance_proxy == "absolute") {
          ts_series <- ts(abs(demeaned))
          method_label <- "Bai-Perron on Absolute Residuals"
        } else {
          stop(sprintf("Unknown variance_proxy: %s", bp_variance_proxy))
        }
      } else {
        ts_series <- ts(series)
        method_label <- "Bai-Perron on Mean"
      }

      bp_candidates <- breakpoints(ts_series ~ 1, h = bp_h, breaks = bp_max_breaks)
      bic_vals <- BIC(bp_candidates)
      best_m <- which.min(bic_vals) - 1
      bp_model <- breakpoints(bp_candidates, breaks = best_m)
      b_pts <- bp_model$breakpoints

      heavy <- list(confint = list(), fstats = list())
      if (bp_compute_heavy_stats && !all(is.na(b_pts))) {
        confint_bp <- confint(bp_model, vcov = kernHAC)
        fstats <- Fstats(ts_series ~ 1, vcov = kernHAC)
        heavy <- list(
          confint = as.data.frame(confint_bp),
          fstats = as.data.frame(fstats$statistic)
        )
      }

      if (all(is.na(b_pts))) {
        structural_breaks_block[[target]] <- list(
          found = FALSE,
          indices = list(),
          dates = list(),
          n_breaks = 0,
          bic = as.numeric(bic_vals),
          confint = heavy$confint,
          fstats = heavy$fstats,
          method = sprintf("%s (HAC, h=%.2f, max_breaks=%d)", method_label, bp_h, bp_max_breaks)
        )
      } else {
        structural_breaks_block[[target]] <- list(
          found = TRUE,
          indices = b_pts,
          dates = as.character(series_dates[b_pts]),
          n_breaks = length(b_pts),
          bic = as.numeric(bic_vals),
          confint = heavy$confint,
          fstats = heavy$fstats,
          method = sprintf("%s (HAC, h=%.2f, max_breaks=%d)", method_label, bp_h, bp_max_breaks)
        )
      }
    }
  } else {
    messages <- c(messages, sprintf("[%s] Structural breaks skipped by config.", col))
  }

  list(
    ok = TRUE,
    col = col,
    result = list(
      adf = adf_block,
      kpss = kpss_block,
      structural_breaks = structural_breaks_block
    ),
    messages = messages
  )
}

process_series_with_retry <- function(col, retries, backoff_seconds) {
  attempt <- 1
  repeat {
    res <- tryCatch(
      process_single_series(col, show_structure = identical(col, ret_cols[1])),
      error = function(e) {
        list(
          ok = FALSE,
          col = col,
          error = e$message,
          messages = c(sprintf("[%s] Attempt %d failed: %s", col, attempt, e$message))
        )
      }
    )

    if (isTRUE(res$ok)) {
      res$attempts <- attempt
      return(res)
    }

    if (attempt > retries) {
      res$attempts <- attempt
      return(res)
    }

    attempt <- attempt + 1
    Sys.sleep(backoff_seconds)
  }
}

series_results <- list()
execution_mode <- "sequential"

if (isTRUE(parallel_enabled) && length(ret_cols) > 1 && parallel_n_workers > 1) {
  execution_mode <- "parallel"
  workers <- min(parallel_n_workers, length(ret_cols))
  if (parallel_n_workers_requested > length(ret_cols)) {
    cat(sprintf("[INFO] Requested %d workers but only %d series available; using %d workers.\n", parallel_n_workers_requested, length(ret_cols), workers))
    flush.console()
  }
  cat(sprintf("--- Parallel mode enabled: %d workers ---\n", workers))
  flush.console()

  cl <- NULL
  parallel_error <- NULL
  tryCatch({
    cl <- parallel::makeCluster(workers)
    parallel::clusterExport(
      cl,
      varlist = c(
        "data", "ret_cols", "run_adf_kpss", "run_structural_breaks",
        "bp_h", "bp_max_breaks", "bp_compute_heavy_stats",
        "bp_targets", "bp_variance_proxy",
        "process_single_series", "process_series_with_retry",
        "max_retries_per_series", "retry_backoff_seconds"
      ),
      envir = environment()
    )
    parallel::clusterEvalQ(cl, {
      .libPaths(c("R_libs", .libPaths()))
      library(urca)
      library(strucchange)
      library(sandwich)
      NULL
    })
    series_results <- parallel::parLapply(
      cl,
      ret_cols,
      function(col) process_series_with_retry(col, max_retries_per_series, retry_backoff_seconds)
    )
  }, error = function(e) {
    parallel_error <<- e$message
  }, finally = {
    if (!is.null(cl)) {
      parallel::stopCluster(cl)
    }
  })

  if (!is.null(parallel_error)) {
    cat(sprintf("[WARN] Parallel execution failed: %s\n", parallel_error))
    flush.console()
    if (isTRUE(fallback_to_sequential_on_parallel_error)) {
      execution_mode <- "sequential_fallback"
      cat("--- Falling back to sequential mode ---\n")
      flush.console()
      series_results <- lapply(
        ret_cols,
        function(col) process_series_with_retry(col, max_retries_per_series, retry_backoff_seconds)
      )
    } else {
      stop(sprintf("Parallel execution failed and fallback is disabled: %s", parallel_error))
    }
  }
} else {
  series_results <- lapply(
    ret_cols,
    function(col) process_series_with_retry(col, max_retries_per_series, retry_backoff_seconds)
  )
}

results[["_meta"]]$execution_mode <- execution_mode

failed_series <- c()
for (res in series_results) {
  col <- res$col
  cat(sprintf("--- Processing %s ---\n", col))
  if (!is.null(res$messages) && length(res$messages) > 0) {
    cat(paste(res$messages, collapse = "\n"), "\n")
  }

  if (isTRUE(res$ok)) {
    results[[col]] <- res$result
    results[["_meta"]]$last_completed_series <- col
  } else {
    results[[col]] <- list(error = res$error)
    results[["_meta"]]$last_error_series <- col
    results[["_meta"]]$last_error_message <- res$error
    failed_series <- c(failed_series, col)
  }

  results[["_meta"]][[paste0("attempts_", col)]] <- res$attempts
  write_json(results, report_path, pretty = TRUE, auto_unbox = TRUE)
  cat(sprintf("[%s] Partial report saved.\n", col))
  flush.console()
}

if (length(failed_series) > 0) {
  results[["_meta"]]$status <- "completed_with_errors"
  results[["_meta"]]$failed_series <- failed_series
} else {
  results[["_meta"]]$status <- "completed"
}
results[["_meta"]]$run_finished <- as.character(Sys.time())
write_json(results, report_path, pretty = TRUE, auto_unbox = TRUE)

cat("--- Diagnostics Completed. Report saved to results/diagnostics/initial_report.json ---\n")
