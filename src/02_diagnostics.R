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

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Config path must be provided as an argument.")
}

config_path <- args[1]
config <- read_yaml(config_path)

# --- CLI Overrides for Orchestration ---
# Usage: Rscript 02_diagnostics.R config.yaml [h] [type: mean|variance] [skip_stats: 0|1] [output_name] [n_workers]
req_h <- if (length(args) >= 2) as.numeric(args[2]) else NA
req_type <- if (length(args) >= 3) args[3] else "mean"
req_skip_stats <- if (length(args) >= 4) (args[4] == "1") else FALSE
req_output_name <- if (length(args) >= 5) args[5] else "initial_report.json"
req_n_workers <- if (length(args) >= 6) as.integer(args[6]) else NA

processed_path <- file.path(config$paths$processed, "fx_aligned.csv")
data <- read_csv(processed_path, show_col_types = FALSE)

# Detect log-return columns
ret_cols <- grep("_RET$", names(data), value = TRUE)

results <- list()

# Section switches
run_adf_kpss <- if (req_skip_stats) FALSE else (if (!is.null(config$diagnostics$run_adf_kpss)) config$diagnostics$run_adf_kpss else TRUE)
run_structural_breaks <- if (!is.null(config$diagnostics$run_structural_breaks)) config$diagnostics$run_structural_breaks else TRUE

# Parallel execution controls
parallel_enabled <- if (!is.null(config$diagnostics$parallel$enabled)) config$diagnostics$parallel$enabled else FALSE
max_retries_per_series <- if (!is.null(config$diagnostics$parallel$max_retries_per_series)) config$diagnostics$parallel$max_retries_per_series else 1
retry_backoff_seconds <- if (!is.null(config$diagnostics$parallel$retry_backoff_seconds)) config$diagnostics$parallel$retry_backoff_seconds else 1
fallback_to_sequential_on_parallel_error <- if (!is.null(config$diagnostics$parallel$fallback_to_sequential_on_parallel_error)) config$diagnostics$parallel$fallback_to_sequential_on_parallel_error else TRUE

available_cores <- parallel::detectCores(logical = TRUE)
if (is.na(available_cores) || available_cores < 1) {
  available_cores <- 2
}

if (!is.na(req_n_workers)) {
  parallel_n_workers_requested <- req_n_workers
  parallel_worker_source <- "cli"
} else if (!is.null(config$diagnostics$parallel$n_workers)) {
  parallel_n_workers_requested <- as.integer(config$diagnostics$parallel$n_workers)
  parallel_worker_source <- "config"
} else {
  parallel_n_workers_requested <- max(1, min(length(ret_cols), max(1, available_cores - 1), 4))
  parallel_worker_source <- "auto"
}

parallel_n_workers <- min(parallel_n_workers_requested, length(ret_cols))

# Bai-Perron runtime controls
bp_h <- if (!is.na(req_h)) req_h else (if (!is.null(config$diagnostics$structural_breaks$h)) config$diagnostics$structural_breaks$h else 0.15)
bp_max_breaks <- if (!is.null(config$diagnostics$structural_breaks$max_breaks)) config$diagnostics$structural_breaks$max_breaks else 3
bp_compute_heavy_stats <- if (!is.null(config$diagnostics$structural_breaks$compute_heavy_stats)) config$diagnostics$structural_breaks$compute_heavy_stats else FALSE

# Prepare output path
out_dir <- file.path(config$paths$results, "diagnostics")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
report_path <- file.path(out_dir, req_output_name)

results[["_meta"]] <- list(
  run_started = as.character(Sys.time()),
  status = "running",
  run_adf_kpss = run_adf_kpss,
  run_structural_breaks = run_structural_breaks,
  parallel_enabled = parallel_enabled,
  parallel_n_workers = parallel_n_workers,
  bp_h = bp_h,
  bp_type = req_type,
  config_path = config_path
)

cat(sprintf("--- Running Diagnostics (h=%.2f, type=%s, skip_stats=%s) ---\n", bp_h, req_type, req_skip_stats))

process_single_series <- function(col, show_structure = FALSE) {
  start_time <- Sys.time()
  valid_idx <- which(!is.na(data[[col]]))
  if (length(valid_idx) == 0) {
    stop(sprintf("No non-NA observations for %s", col))
  }

  series <- data[[col]][valid_idx]
  # Variance logic: use squared returns
  if (req_type == "variance") {
    series <- series^2
  }

  series_dates <- data$Date[valid_idx]
  messages <- c()

  adf_block <- list(status = "skipped")
  kpss_block <- list(status = "skipped")
  structural_breaks_block <- list(found = NA, n_breaks = NA, method = "skipped")

  if (run_adf_kpss) {
    adf <- ur.df(series, type = "drift", selectlags = "AIC")
    adf_stat <- if ("tau2" %in% names(adf@teststat)) adf@teststat["tau2"] else adf@teststat[[1]]
    adf_cval <- if (!is.null(dimnames(adf@cval)[[1]]) && "tau2" %in% dimnames(adf@cval)[[1]]) adf@cval["tau2",] else adf@cval[1,]
    
    kpss <- ur.kpss(series, type = "mu", lags = "short")
    kpss_stat <- kpss@teststat
    kpss_cval <- kpss@cval
    if (is.null(names(kpss_cval))) names(kpss_cval) <- c("10pct", "5pct", "2.5pct", "1pct")

    adf_block <- list(statistic = adf_stat, critical_values = adf_cval, decision = if (adf_stat < adf_cval["5pct"]) "Stationary" else "Non-Stationary")
    kpss_block <- list(statistic = kpss_stat, critical_values = kpss_cval, decision = if (kpss_stat > kpss_cval["5pct"]) "Non-Stationary" else "Stationary")
  }

  if (run_structural_breaks) {
    messages <- c(messages, sprintf("[%s] Bai-Perron (h=%.2f)...", col, bp_h))
    ts_series <- ts(series)
    bp_candidates <- breakpoints(ts_series ~ 1, h = bp_h, breaks = bp_max_breaks)
    bic_vals <- BIC(bp_candidates)
    best_m <- which.min(bic_vals) - 1
    bp_model <- breakpoints(bp_candidates, breaks = best_m)
    b_pts <- bp_model$breakpoints

    heavy <- list(confint = list(), fstats = list())
    if (bp_compute_heavy_stats && !all(is.na(b_pts))) {
      confint_bp <- confint(bp_model, vcov = kernHAC)
      fstats <- Fstats(ts_series ~ 1, vcov = kernHAC)
      heavy <- list(confint = as.data.frame(confint_bp), fstats = as.data.frame(fstats$statistic))
    }

    structural_breaks_block <- list(
      found = !all(is.na(b_pts)),
      indices = if (all(is.na(b_pts))) list() else b_pts,
      dates = if (all(is.na(b_pts))) list() else as.character(series_dates[b_pts]),
      n_breaks = if (all(is.na(b_pts))) 0 else length(b_pts),
      bic = as.numeric(bic_vals),
      confint = heavy$confint,
      fstats = heavy$fstats,
      method = sprintf("Bai-Perron (%s, h=%.2f)", req_type, bp_h)
    )
  }

  end_time <- Sys.time()
  duration_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

  list(
    ok = TRUE,
    col = col,
    duration_secs = duration_secs,
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
      process_single_series(col),
      error = function(e) list(ok = FALSE, col = col, error = e$message, messages = c(sprintf("[%s] Error: %s", col, e$message)))
    )
    if (isTRUE(res$ok) || attempt > retries) return(res)
    attempt <- attempt + 1
    Sys.sleep(backoff_seconds)
  }
}

series_results <- list()
if (isTRUE(parallel_enabled) && length(ret_cols) > 1 && parallel_n_workers > 1) {
  cl <- parallel::makeCluster(parallel_n_workers)
  tryCatch({
    parallel::clusterExport(cl, varlist = c("data", "ret_cols", "run_adf_kpss", "run_structural_breaks", "bp_h", "bp_max_breaks", "bp_compute_heavy_stats", "process_single_series", "req_type"), envir = environment())
    parallel::clusterEvalQ(cl, { .libPaths(c("R_libs", .libPaths())); library(urca); library(strucchange); NULL })
    series_results <- parallel::parLapply(cl, ret_cols, function(col) process_single_series(col))
  }, finally = parallel::stopCluster(cl))
} else {
  series_results <- lapply(ret_cols, function(col) process_single_series(col))
}

for (res in series_results) {
  results[[res$col]] <- res$result
  results[["_meta"]]$durations[[res$col]] <- res$duration_secs
}

results[["_meta"]]$status <- "completed"
results[["_meta"]]$run_finished <- as.character(Sys.time())
write_json(results, report_path, pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("--- Completed. Report: %s ---\n", report_path))

