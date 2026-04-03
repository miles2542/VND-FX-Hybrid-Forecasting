# 06_exogenous_diagnostics.R - Exogenous Diagnostics for ARIMAX/VARX prep
# Purpose:
# 1) Detect structural breaks (Bai-Perron) in macro exogenous variables.
# 2) Run CCF between stationary macro exogenous series and FX return targets.

.libPaths(c("R_libs", .libPaths()))

library(readr)
library(yaml)
library(strucchange)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Config path must be provided as an argument.")
}

config_path <- args[1]
config <- read_yaml(config_path)

raw_dir <- config$paths$raw
processed_dir <- config$paths$processed
results_dir <- config$paths$results

# Reuse Bai-Perron controls from diagnostics config for consistency with 02_diagnostics.R
bp_h <- if (!is.null(config$diagnostics$structural_breaks$h)) config$diagnostics$structural_breaks$h else 0.15
bp_max_breaks <- if (!is.null(config$diagnostics$structural_breaks$max_breaks)) config$diagnostics$structural_breaks$max_breaks else 3

out_dir <- file.path(results_dir, "diagnostics", "exogenous")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# -----------------------------
# 1) Load target FX returns (Y)
# -----------------------------
y_path <- file.path(processed_dir, "fx_aligned.csv")
y_df <- read_csv(y_path, show_col_types = FALSE)

if (!("Date" %in% names(y_df))) {
  stop("fx_aligned.csv must contain a Date column.")
}

y_df$Date <- as.Date(y_df$Date)
ret_cols <- grep("_RET$", names(y_df), value = TRUE)
if (length(ret_cols) == 0) {
  stop("No target FX return columns found in fx_aligned.csv (expected *_RET columns).")
}

# -----------------------------
# 2) Load 4 macro exogenous (X) from data/raw
# -----------------------------
read_macro_series <- function(path, std_name) {
  df <- read_csv(path, show_col_types = FALSE)

  if (!("Date" %in% names(df))) {
    names(df)[1] <- "Date"
  }
  df$Date <- as.Date(df$Date)

  value_cols <- setdiff(names(df), "Date")
  if (length(value_cols) != 1) {
    stop(sprintf("%s must have exactly one value column besides Date; got: %s", path, paste(value_cols, collapse = ", ")))
  }

  names(df)[names(df) == value_cols[1]] <- std_name
  df[[std_name]] <- as.numeric(df[[std_name]])
  df[, c("Date", std_name)]
}

x_dff <- read_macro_series(file.path(raw_dir, "DFF.csv"), "DFF")
x_dfx <- read_macro_series(file.path(raw_dir, "DFX.csv"), "DFX")
x_gold <- read_macro_series(file.path(raw_dir, "Gold.csv"), "Gold")
x_sbv <- read_macro_series(file.path(raw_dir, "sbv_refi_daily.csv"), "SBV_Refi")

# -----------------------------
# 3) Secure date alignment: outer merge + sort + ffill
# -----------------------------
x_merged <- Reduce(
  function(a, b) merge(a, b, by = "Date", all = TRUE),
  list(x_dff, x_dfx, x_gold, x_sbv)
)
x_merged <- x_merged[order(x_merged$Date), ]

# Forward-fill helper (base R)
ffill_vec <- function(v) {
  idx <- which(!is.na(v))
  if (length(idx) == 0) return(v)
  for (i in seq_len(length(v))) {
    if (is.na(v[i]) && i > 1) v[i] <- v[i - 1]
  }
  v
}

for (col in c("DFF", "DFX", "Gold", "SBV_Refi")) {
  x_merged[[col]] <- ffill_vec(x_merged[[col]])
}

# Remove leading rows where at least one series remains NA after ffill
x_merged <- x_merged[stats::complete.cases(x_merged[, c("DFF", "DFX", "Gold", "SBV_Refi")]), ]

# -----------------------------
# 4) Bai-Perron structural breaks on macro X (same methodology)
# -----------------------------
run_bai_perron <- function(series, dates, h, max_breaks) {
  ts_series <- ts(series)
  bp_candidates <- breakpoints(ts_series ~ 1, h = h, breaks = max_breaks)
  bic_vals <- BIC(bp_candidates)
  best_m <- which.min(bic_vals) - 1
  bp_model <- breakpoints(bp_candidates, breaks = best_m)
  b_pts <- bp_model$breakpoints

  if (all(is.na(b_pts))) {
    list(
      found = FALSE,
      indices = list(),
      dates = list(),
      n_breaks = 0,
      bic = as.numeric(bic_vals),
      method = sprintf("Bai-Perron (h=%.2f, max_breaks=%d)", h, max_breaks)
    )
  } else {
    list(
      found = TRUE,
      indices = as.numeric(b_pts),
      dates = as.character(dates[b_pts]),
      n_breaks = length(b_pts),
      bic = as.numeric(bic_vals),
      method = sprintf("Bai-Perron (h=%.2f, max_breaks=%d)", h, max_breaks)
    )
  }
}

break_results <- list(
  DFF = run_bai_perron(x_merged$DFF, x_merged$Date, bp_h, bp_max_breaks),
  DFX = run_bai_perron(x_merged$DFX, x_merged$Date, bp_h, bp_max_breaks),
  Gold = run_bai_perron(x_merged$Gold, x_merged$Date, bp_h, bp_max_breaks),
  SBV_Refi = run_bai_perron(x_merged$SBV_Refi, x_merged$Date, bp_h, bp_max_breaks)
)

cat("--- Bai-Perron Structural Breaks (Exogenous X) ---\n")
for (nm in names(break_results)) {
  br <- break_results[[nm]]
  if (isTRUE(br$found)) {
    cat(sprintf("[%s] Break dates: %s\n", nm, paste(br$dates, collapse = ", ")))
  } else {
    cat(sprintf("[%s] No breaks selected by BIC.\n", nm))
  }
}

# -----------------------------
# 5) Stationary transforms for CCF
# -----------------------------
x_stationary <- data.frame(
  Date = x_merged$Date,
  DFF_diff = c(NA, diff(x_merged$DFF)),
  SBV_Refi_diff = c(NA, diff(x_merged$SBV_Refi)),
  DFX_logret = c(NA, 100 * diff(log(x_merged$DFX))),
  Gold_logret = c(NA, 100 * diff(log(x_merged$Gold)))
)
x_stationary <- x_stationary[stats::complete.cases(x_stationary), ]

# -----------------------------
# 6) CCF: stationary X vs FX returns Y
# -----------------------------
ccf_summary <- list()

pdf(file.path(out_dir, "ccf_plots.pdf"), width = 10, height = 7)
for (y_col in ret_cols) {
  y_sub <- y_df[, c("Date", y_col)]
  names(y_sub) <- c("Date", "Y")

  merged_xy <- merge(x_stationary, y_sub, by = "Date", all = FALSE)
  merged_xy <- merged_xy[stats::complete.cases(merged_xy), ]

  if (nrow(merged_xy) < 30) {
    ccf_summary[[y_col]] <- list(error = "Too few aligned observations for CCF")
    next
  }

  cat(sprintf("\n--- CCF vs %s ---\n", y_col))
  ccf_summary[[y_col]] <- list()

  for (x_col in c("DFF_diff", "SBV_Refi_diff", "DFX_logret", "Gold_logret")) {
    cc <- ccf(merged_xy[[x_col]], merged_xy$Y, lag.max = 30, plot = TRUE,
              main = sprintf("CCF: %s vs %s", x_col, y_col), na.action = na.omit)

    peak_idx <- which.max(abs(cc$acf))
    peak_lag <- as.numeric(cc$lag[peak_idx])
    peak_corr <- as.numeric(cc$acf[peak_idx])

    ccf_summary[[y_col]][[x_col]] <- list(
      peak_lag = peak_lag,
      peak_corr = peak_corr,
      n_obs = nrow(merged_xy)
    )

    cat(sprintf("%s -> peak |corr| at lag %d: %.4f (n=%d)\n", x_col, peak_lag, peak_corr, nrow(merged_xy)))
  }
}
dev.off()

# Persist outputs
write_csv(x_merged, file.path(out_dir, "exog_merged_ffill.csv"))
write_csv(x_stationary, file.path(out_dir, "exog_stationary_for_ccf.csv"))
write_json(
  list(
    meta = list(
      run_time = as.character(Sys.time()),
      config_path = config_path,
      bp_h = bp_h,
      bp_max_breaks = bp_max_breaks
    ),
    structural_breaks = break_results,
    ccf_summary = ccf_summary
  ),
  file.path(out_dir, "exogenous_diagnostics_report.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

cat("\n--- Exogenous diagnostics completed. ---\n")
cat(sprintf("Outputs saved in: %s\n", out_dir))
cat("- exog_merged_ffill.csv\n")
cat("- exog_stationary_for_ccf.csv\n")
cat("- ccf_plots.pdf\n")
cat("- exogenous_diagnostics_report.json\n")
