# 03_parametric.R - ARIMA & VAR Stage
# Lead role: Implement baseline parametric models (Zhang 2003 residual hybrid start)

# Purely relative path for R_libs
.libPaths(c("R_libs", .libPaths()))

library(forecast)
library(vars)
library(jsonlite)
library(readr)
library(yaml)
library(urca)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Config path must be provided as an argument.")
}

config <- read_yaml(args[1])
processed_dir <- file.path(config$paths$processed, config$active_target)
results_dir <- file.path(config$paths$results, config$active_target)

# 1. Load Data
train_df <- read_csv(file.path(processed_dir, "train.csv"), show_col_types = FALSE)
val_df <- read_csv(file.path(processed_dir, "val.csv"), show_col_types = FALSE)
test_df <- read_csv(file.path(processed_dir, "test.csv"), show_col_types = FALSE)

exog_df <- read_csv(file.path(processed_dir, "exogenous_php.csv"), show_col_types = FALSE)
date_col <- names(train_df)[1]

if (!("Date" %in% names(exog_df))) {
  stop("exogenous_php.csv must contain a Date column.")
}

if (!(date_col %in% names(train_df)) || !(date_col %in% names(val_df)) || !(date_col %in% names(test_df))) {
  stop("Date column is missing from one or more split files.")
}

exog_df$Date <- as.character(exog_df$Date)
fx_dates_chr <- as.character(c(train_df[[date_col]], val_df[[date_col]], test_df[[date_col]]))

# Align exogenous rows to exact FX split dates; unmatched rows become NA and are filled via LOCF.
idx <- match(fx_dates_chr, exog_df$Date)
aligned_exog <- exog_df[idx, , drop = FALSE]
aligned_exog$Date <- fx_dates_chr

exog_cols <- setdiff(names(aligned_exog), "Date")
for (col_name in exog_cols) {
  x <- as.numeric(aligned_exog[[col_name]])
  x <- zoo::na.locf(x, na.rm = FALSE)
  x <- zoo::na.locf(x, fromLast = TRUE, na.rm = FALSE)
  if (all(is.na(x))) {
    stop(sprintf("Exogenous column %s is entirely NA after alignment/fill.", col_name))
  }
  aligned_exog[[col_name]] <- x
}

n_train <- nrow(train_df)
n_val <- nrow(val_df)
n_test <- nrow(test_df)

train_exog <- as.matrix(aligned_exog[seq_len(n_train), exog_cols, drop = FALSE])
val_exog <- as.matrix(aligned_exog[(n_train + 1):(n_train + n_val), exog_cols, drop = FALSE])
test_exog <- as.matrix(aligned_exog[(n_train + n_val + 1):(n_train + n_val + n_test), exog_cols, drop = FALSE])
all_exog_mat <- as.matrix(aligned_exog[, exog_cols, drop = FALSE])

storage.mode(train_exog) <- "numeric"
storage.mode(val_exog) <- "numeric"
storage.mode(test_exog) <- "numeric"
storage.mode(all_exog_mat) <- "numeric"

ret_cols <- grep("_RET$", names(train_df), value = TRUE)

# ------------------------------------------------------------------------------
# STAGE 1: ARIMA (Univariate)
# ------------------------------------------------------------------------------
cat("\n--- Fitting ARIMA Models ---\n")
arima_results_dir <- file.path(results_dir, "arima")
if (!dir.exists(arima_results_dir)) dir.create(arima_results_dir, recursive = TRUE)

arima_forecasts <- list()
arima_diag <- list()
arima_residuals_train <- data.frame(Date = train_df[[1]])
arima_params <- list()

for (col in ret_cols) {
  cat(sprintf("Fitting auto.arima for %s...\n", col))
  
  # Fit on training data
  # Paper rationale: Pure AR/MA selection
  train_series <- train_df[[col]]
  fit <- auto.arima(train_series, 
                    max.p = config$arima$max_p, 
                    max.q = config$arima$max_q, 
                    max.d = config$arima$max_d,
                    seasonal = config$arima$seasonal,
                    ic = config$arima$ic,
                    stepwise = FALSE) # More thorough search for academia
  
  # 1-Step Rolling Forecast (Academic Corrective)
  # ---------------------------------------------
  # We pass the entire dataset through the KALMAN FILTER of the fitted model
  # to get the 1-step ahead prediction (E[Y_t | I_{t-1}]) for every point.
  all_series <- c(train_series, val_df[[col]], test_df[[col]])
  fit_eval <- Arima(all_series, model = fit)
  
  fitted_vals <- as.numeric(fitted(fit_eval))
  resid_vals <- as.numeric(residuals(fit_eval))
  
  # Diagnostics (Training only)
  arima_diag[[col]] <- list(
    order = as.numeric(arimaorder(fit)),
    aic = AIC(fit),
    bic = BIC(fit)
  )

  # Parameter persistence for downstream t/z testing
  coef_vals <- fit$coef
  se_vals <- rep(NA_real_, length(coef_vals))
  if (!is.null(fit$var.coef)) {
    se_vals <- suppressWarnings(sqrt(diag(fit$var.coef)))
  }
  # Fallback when var.coef is unavailable/degenerate
  if (any(is.na(se_vals)) || any(!is.finite(se_vals))) {
    fit_sum <- summary(fit)
    if (!is.null(fit_sum$coef)) {
      coef_tab <- fit_sum$coef
      # summary matrix columns are typically: Estimate, s.e., z value, Pr(>|z|)
      if (ncol(coef_tab) >= 2) {
        se_from_sum <- as.numeric(coef_tab[, 2])
        names(se_from_sum) <- rownames(coef_tab)
        for (k in names(coef_vals)) {
          if (k %in% names(se_from_sum) && (is.na(se_vals[k]) || !is.finite(se_vals[k]))) {
            se_vals[k] <- se_from_sum[k]
          }
        }
      }
    }
  }
  se_vals[se_vals <= 0] <- NA_real_
  t_vals <- coef_vals / se_vals
  p_vals <- 2 * pnorm(abs(t_vals), lower.tail = FALSE)
  arima_params[[col]] <- data.frame(
    Pair = col,
    Parameter = names(coef_vals),
    Estimate = as.numeric(coef_vals),
    `Std. Error` = as.numeric(se_vals),
    `t-value` = as.numeric(t_vals),
    `P-value` = as.numeric(p_vals),
    check.names = FALSE
  )
  
  # Residuals on training set (needed for SVR/MLP)
  arima_residuals_train[[col]] <- resid_vals[seq_len(nrow(train_df))]
  
  # Extract VAL and TEST portions
  n_train <- nrow(train_df)
  n_val <- nrow(val_df)
  n_test <- nrow(test_df)
  
  val_indices <- (n_train + 1):(n_train + n_val)
  test_indices <- (n_train + n_val + 1):(n_train + n_val + n_test)
  
  # Combine actual and forecast for reporting
  pair_fc <- data.frame(
    Date = c(val_df[[1]], test_df[[1]]),
    Actual = c(val_df[[col]], test_df[[col]]),
    Forecast = fitted_vals[c(val_indices, test_indices)],
    Set = c(rep("val", n_val), rep("test", n_test))
  )
  pair_fc$Pair <- col
  arima_forecasts[[col]] <- pair_fc
}

# Save ARIMA outputs
write_json(arima_diag, file.path(arima_results_dir, "diagnostics.json"), pretty = TRUE, auto_unbox = TRUE)
write_csv(do.call(rbind, arima_forecasts), file.path(arima_results_dir, "forecasts.csv"))
write_csv(arima_residuals_train, file.path(arima_results_dir, "residuals_train.csv"))
write_csv(do.call(rbind, arima_params), file.path(arima_results_dir, "parameters.csv"))

# ------------------------------------------------------------------------------
# STAGE 1B: ARIMAX (Univariate + Exogenous)
# ------------------------------------------------------------------------------
cat("\n--- Fitting ARIMAX Models ---\n")
arimax_results_dir <- file.path(results_dir, "arimax")
if (!dir.exists(arimax_results_dir)) dir.create(arimax_results_dir, recursive = TRUE)

arimax_forecasts <- list()
arimax_diag <- list()
arimax_residuals_train <- data.frame(Date = train_df[[1]])
arimax_params <- list()

for (col in ret_cols) {
  cat(sprintf("Fitting auto.arima (xreg) for %s...\n", col))
  
  train_series <- train_df[[col]]
  fit <- auto.arima(train_series,
                    xreg = train_exog,
                    max.p = config$arima$max_p,
                    max.q = config$arima$max_q,
                    max.d = config$arima$max_d,
                    seasonal = config$arima$seasonal,
                    ic = config$arima$ic,
                    stepwise = FALSE)
  
  all_series <- c(train_series, val_df[[col]], test_df[[col]])
  fit_eval <- Arima(all_series, model = fit, xreg = all_exog_mat)
  
  fitted_vals <- as.numeric(fitted(fit_eval))
  resid_vals <- as.numeric(residuals(fit_eval))
  
  arimax_diag[[col]] <- list(
    order = as.numeric(arimaorder(fit)),
    aic = AIC(fit),
    bic = BIC(fit)
  )

  # Parameter persistence for downstream t/z testing
  coef_vals <- fit$coef
  se_vals <- rep(NA_real_, length(coef_vals))
  if (!is.null(fit$var.coef)) {
    se_vals <- suppressWarnings(sqrt(diag(fit$var.coef)))
  }
  # Fallback when var.coef is unavailable/degenerate
  if (any(is.na(se_vals)) || any(!is.finite(se_vals))) {
    fit_sum <- summary(fit)
    if (!is.null(fit_sum$coef)) {
      coef_tab <- fit_sum$coef
      if (ncol(coef_tab) >= 2) {
        se_from_sum <- as.numeric(coef_tab[, 2])
        names(se_from_sum) <- rownames(coef_tab)
        for (k in names(coef_vals)) {
          if (k %in% names(se_from_sum) && (is.na(se_vals[k]) || !is.finite(se_vals[k]))) {
            se_vals[k] <- se_from_sum[k]
          }
        }
      }
    }
  }
  se_vals[se_vals <= 0] <- NA_real_
  t_vals <- coef_vals / se_vals
  p_vals <- 2 * pnorm(abs(t_vals), lower.tail = FALSE)
  arimax_params[[col]] <- data.frame(
    Pair = col,
    Parameter = names(coef_vals),
    Estimate = as.numeric(coef_vals),
    `Std. Error` = as.numeric(se_vals),
    `t-value` = as.numeric(t_vals),
    `P-value` = as.numeric(p_vals),
    check.names = FALSE
  )
  
  arimax_residuals_train[[col]] <- resid_vals[seq_len(nrow(train_df))]
  
  n_train <- nrow(train_df)
  n_val <- nrow(val_df)
  n_test <- nrow(test_df)
  
  val_indices <- (n_train + 1):(n_train + n_val)
  test_indices <- (n_train + n_val + 1):(n_train + n_val + n_test)
  
  pair_fc <- data.frame(
    Date = c(val_df[[1]], test_df[[1]]),
    Actual = c(val_df[[col]], test_df[[col]]),
    Forecast = fitted_vals[c(val_indices, test_indices)],
    Set = c(rep("val", n_val), rep("test", n_test))
  )
  pair_fc$Pair <- col
  arimax_forecasts[[col]] <- pair_fc
}

write_json(arimax_diag, file.path(arimax_results_dir, "diagnostics.json"), pretty = TRUE, auto_unbox = TRUE)
write_csv(do.call(rbind, arimax_forecasts), file.path(arimax_results_dir, "forecasts.csv"))
write_csv(arimax_residuals_train, file.path(arimax_results_dir, "residuals_train.csv"))
write_csv(do.call(rbind, arimax_params), file.path(arimax_results_dir, "parameters.csv"))

# ------------------------------------------------------------------------------
# STAGE 2: VAR (Multivariate)
# ------------------------------------------------------------------------------
cat("\n--- Fitting VAR Model ---\n")
var_results_dir <- file.path(results_dir, "var")
if (!dir.exists(var_results_dir)) dir.create(var_results_dir, recursive = TRUE)

# Joint stationarity - Johansen Test (as per task chunk)
var_data_train <- as.matrix(train_df[, ret_cols])
johansen_test <- ca.jo(var_data_train, type = "trace", ecdet = "const", spec = "transitory")

# VAR Lag selection
lag_select <- VARselect(var_data_train, lag.max = config$var$max_lags, type = "both")
# Match the IC name from config (e.g., "AIC", "SC", "HQ", "FPE")
# Note: vars package uses "SC(n)", "AIC(n)", etc.
ic_to_use <- toupper(config$var$ic)
matched_name <- names(lag_select$selection)[grep(ic_to_use, names(lag_select$selection))]

if (length(matched_name) > 0) {
  selected_p <- as.numeric(lag_select$selection[matched_name[1]])
} else {
  # Fallback to first available criterion if match fails
  selected_p <- as.numeric(lag_select$selection[1])
  ic_to_use <- names(lag_select$selection)[1]
}

if (is.na(selected_p) || selected_p < 1) selected_p <- 1
cat(sprintf("Selected VAR lag (p) via %s: %d\n", ic_to_use, selected_p))

var_fit <- VAR(var_data_train, p = selected_p, type = "both")

var_params <- list()
for (eq_name in names(var_fit$varresult)) {
  eq_summary <- summary(var_fit$varresult[[eq_name]])
  coef_tab <- coef(eq_summary)
  var_params[[eq_name]] <- data.frame(
    Equation = eq_name,
    Parameter = rownames(coef_tab),
    Estimate = as.numeric(coef_tab[, "Estimate"]),
    `Std. Error` = as.numeric(coef_tab[, "Std. Error"]),
    `t-value` = as.numeric(coef_tab[, "t value"]),
    `P-value` = as.numeric(coef_tab[, "Pr(>|t|)"]),
    check.names = FALSE
  )
}

# 1-Step Rolling Forecast (Manual Coefficient Application)
# ---------------------------------------------------------
# Concatenate all data
all_data_mat <- as.matrix(rbind(train_df[, ret_cols], val_df[, ret_cols], test_df[, ret_cols]))
n_total <- nrow(all_data_mat)

# Prepare matrices for 1-step calculation: Y_t = C + A1*Y_{t-1} + ... + Ap*Y_{t-p}
# We use fitted() values for the entire series
var_fitted_vals <- matrix(NA, nrow = n_total, ncol = length(ret_cols))
colnames(var_fitted_vals) <- ret_cols

# The VAR model already has $fitted.values for the training portion (starting at p+1)
fitted_train <- fitted(var_fit)
var_fitted_vals[(selected_p + 1):nrow(train_df), ] <- fitted_train

# For Validation and Test, we manually project 1-step ahead at each t
# using the realized values at t-1...t-p
coef_list <- coef(var_fit)

for (t in (nrow(train_df) + 1):n_total) {
  for (i in seq_along(ret_cols)) {
    target_col <- ret_cols[i]
    c_vals <- coef_list[[target_col]]
    
    # Calculate: Const + Trend (if any) + Lagged components
    # c_vals structure row names: [pair1.l1, pair2.l1, ..., pairN.lp, const, trend]
    pred_val <- 0
    
    # 1. Add Lagged Components (Name-Matched)
    for (p_idx in 1:selected_p) {
      for (v_col in ret_cols) {
        row_name <- paste0(v_col, ".l", p_idx)
        if (row_name %in% rownames(c_vals)) {
          pred_val <- pred_val + c_vals[row_name, 1] * all_data_mat[t - p_idx, v_col]
        }
      }
    }
    
    # 2. Add Const/Trend (Name-Matched)
    if ("const" %in% rownames(c_vals)) pred_val <- pred_val + c_vals["const", 1]
    if ("trend" %in% rownames(c_vals)) pred_val <- pred_val + c_vals["trend", 1]
    
    var_fitted_vals[t, i] <- pred_val
  }
}

# Organize and Save
var_forecasts_list <- list()
var_residuals_train <- data.frame(Date = train_df[[1]])
all_res_mat <- all_data_mat - var_fitted_vals

for (i in seq_along(ret_cols)) {
  col <- ret_cols[i]
  
  # Training residuals (padding the dead zone with NA or 0)
  # We use 0 for the dead zone to keep logic simple in the MLP features
  res_train <- all_res_mat[seq_len(nrow(train_df)), i]
  res_train[is.na(res_train)] <- 0
  var_residuals_train[[col]] <- res_train
  
  # Forecasts for Val/Test
  val_indices <- (nrow(train_df) + 1):(nrow(train_df) + nrow(val_df))
  test_indices <- (nrow(train_df) + nrow(val_df) + 1):n_total
  
  pair_fc <- data.frame(
    Date = c(val_df$Date, test_df$Date),
    Actual = c(val_df[[col]], test_df[[col]]),
    Forecast = var_fitted_vals[c(val_indices, test_indices), i],
    Set = c(rep("val", nrow(val_df)), rep("test", nrow(test_df))),
    Pair = col
  )
  var_forecasts_list[[col]] <- pair_fc
}

# Save VAR outputs
var_diag <- list(
  selected_lag = as.numeric(selected_p),
  aic = AIC(var_fit),
  johansen = list(
    test_stat = as.numeric(johansen_test@teststat),
    critical_vals = johansen_test@cval
  )
)

write_json(var_diag, file.path(var_results_dir, "diagnostics.json"), pretty = TRUE, auto_unbox = TRUE)
write_csv(do.call(rbind, var_forecasts_list), file.path(var_results_dir, "forecasts.csv"))
write_csv(var_residuals_train, file.path(var_results_dir, "residuals_train.csv"))
write_csv(do.call(rbind, var_params), file.path(var_results_dir, "parameters.csv"))

# ------------------------------------------------------------------------------
# STAGE 2B: VARX (Multivariate + Exogenous)
# ------------------------------------------------------------------------------
cat("\n--- Fitting VARX Model ---\n")
varx_results_dir <- file.path(results_dir, "varx")
if (!dir.exists(varx_results_dir)) dir.create(varx_results_dir, recursive = TRUE)

varx_data_train <- as.matrix(train_df[, ret_cols])
varx_johansen_test <- ca.jo(varx_data_train, type = "trace", ecdet = "const", spec = "transitory")

varx_lag_select <- VARselect(varx_data_train, lag.max = config$var$max_lags, type = "both", exogen = train_exog)
varx_ic_to_use <- toupper(config$var$ic)
varx_matched_name <- names(varx_lag_select$selection)[grep(varx_ic_to_use, names(varx_lag_select$selection))]

if (length(varx_matched_name) > 0) {
  varx_selected_p <- as.numeric(varx_lag_select$selection[varx_matched_name[1]])
} else {
  varx_selected_p <- as.numeric(varx_lag_select$selection[1])
  varx_ic_to_use <- names(varx_lag_select$selection)[1]
}

if (is.na(varx_selected_p) || varx_selected_p < 1) varx_selected_p <- 1
cat(sprintf("Selected VARX lag (p) via %s: %d\n", varx_ic_to_use, varx_selected_p))

varx_fit <- VAR(varx_data_train, p = varx_selected_p, type = "both", exogen = train_exog)

varx_params <- list()
for (eq_name in names(varx_fit$varresult)) {
  eq_summary <- summary(varx_fit$varresult[[eq_name]])
  coef_tab <- coef(eq_summary)
  varx_params[[eq_name]] <- data.frame(
    Equation = eq_name,
    Parameter = rownames(coef_tab),
    Estimate = as.numeric(coef_tab[, "Estimate"]),
    `Std. Error` = as.numeric(coef_tab[, "Std. Error"]),
    `t-value` = as.numeric(coef_tab[, "t value"]),
    `P-value` = as.numeric(coef_tab[, "Pr(>|t|)"]),
    check.names = FALSE
  )
}

all_data_mat <- as.matrix(rbind(train_df[, ret_cols], val_df[, ret_cols], test_df[, ret_cols]))
n_total <- nrow(all_data_mat)

varx_fitted_vals <- matrix(NA, nrow = n_total, ncol = length(ret_cols))
colnames(varx_fitted_vals) <- ret_cols

varx_fitted_train <- fitted(varx_fit)
varx_fitted_vals[(varx_selected_p + 1):nrow(train_df), ] <- varx_fitted_train

coef_list <- coef(varx_fit)
exog_col_names <- colnames(train_exog)

for (t in (nrow(train_df) + 1):n_total) {
  for (i in seq_along(ret_cols)) {
    target_col <- ret_cols[i]
    c_vals <- coef_list[[target_col]]

    pred_val <- 0

    for (p_idx in 1:varx_selected_p) {
      for (v_col in ret_cols) {
        row_name <- paste0(v_col, ".l", p_idx)
        if (row_name %in% rownames(c_vals)) {
          pred_val <- pred_val + c_vals[row_name, 1] * all_data_mat[t - p_idx, v_col]
        }
      }
    }

    if ("const" %in% rownames(c_vals)) pred_val <- pred_val + c_vals["const", 1]
    if ("trend" %in% rownames(c_vals)) pred_val <- pred_val + c_vals["trend", 1]

    # Add exogenous term: beta_j * X_{t,j} for all exogenous columns.
    for (ex_col in exog_col_names) {
      if (ex_col %in% rownames(c_vals)) {
        pred_val <- pred_val + c_vals[ex_col, 1] * all_exog_mat[t, ex_col]
      } else {
        exo_name <- paste0("exo_", ex_col)
        if (exo_name %in% rownames(c_vals)) {
          pred_val <- pred_val + c_vals[exo_name, 1] * all_exog_mat[t, ex_col]
        }
      }
    }

    varx_fitted_vals[t, i] <- pred_val
  }
}

varx_forecasts_list <- list()
varx_residuals_train <- data.frame(Date = train_df[[1]])
varx_all_res_mat <- all_data_mat - varx_fitted_vals

for (i in seq_along(ret_cols)) {
  col <- ret_cols[i]

  res_train <- varx_all_res_mat[seq_len(nrow(train_df)), i]
  res_train[is.na(res_train)] <- 0
  varx_residuals_train[[col]] <- res_train

  val_indices <- (nrow(train_df) + 1):(nrow(train_df) + nrow(val_df))
  test_indices <- (nrow(train_df) + nrow(val_df) + 1):n_total

  pair_fc <- data.frame(
    Date = c(val_df$Date, test_df$Date),
    Actual = c(val_df[[col]], test_df[[col]]),
    Forecast = varx_fitted_vals[c(val_indices, test_indices), i],
    Set = c(rep("val", nrow(val_df)), rep("test", nrow(test_df))),
    Pair = col
  )
  varx_forecasts_list[[col]] <- pair_fc
}

varx_diag <- list(
  selected_lag = as.numeric(varx_selected_p),
  aic = AIC(varx_fit),
  johansen = list(
    test_stat = as.numeric(varx_johansen_test@teststat),
    critical_vals = varx_johansen_test@cval
  )
)

write_json(varx_diag, file.path(varx_results_dir, "diagnostics.json"), pretty = TRUE, auto_unbox = TRUE)
write_csv(do.call(rbind, varx_forecasts_list), file.path(varx_results_dir, "forecasts.csv"))
write_csv(varx_residuals_train, file.path(varx_results_dir, "residuals_train.csv"))
write_csv(do.call(rbind, varx_params), file.path(varx_results_dir, "parameters.csv"))

cat("\n--- Parametric Stage Completed. Results saved to results/arima/, results/arimax/, results/var/, and results/varx/ ---\n")

# ------------------------------------------------------------------------------
# STAGE 3: Baselines (Academic Benchmarks)
# ------------------------------------------------------------------------------
run_baselines <- if (!is.null(config$baselines$enabled)) config$baselines$enabled else TRUE
baseline_models <- if (!is.null(config$baselines$models)) config$baselines$models else c("rw", "mean", "ar1")

if (run_baselines) {
  cat("\n--- Generating Baseline Models ---\n")

  baseline_rw_list <- list()
  baseline_mean_list <- list()
  baseline_ar1_list <- list()

  for (col in ret_cols) {
    cat(sprintf("Computing baselines for %s...\n", col))
    
    n_train_loc <- nrow(train_df)
    n_val_loc <- nrow(val_df)
    n_test_loc <- nrow(test_df)
    actuals <- c(val_df[[col]], test_df[[col]])
    dates <- c(val_df$Date, test_df$Date)
    sets <- c(rep("val", n_val_loc), rep("test", n_test_loc))
    
    # 1. Naive Random Walk (Zero Forecast in Log-Returns)
    if ("rw" %in% baseline_models) {
      baseline_rw_list[[col]] <- data.frame(
        Date = dates, Actual = actuals, Forecast = 0, Set = sets, Pair = col
      )
    }
    
    # 2. Historical Mean
    if ("mean" %in% baseline_models) {
      train_mean <- mean(train_df[[col]], na.rm = TRUE)
      baseline_mean_list[[col]] <- data.frame(
        Date = dates, Actual = actuals, Forecast = train_mean, Set = sets, Pair = col
      )
    }
    
    # 3. AR(1) OLS: r_t = b0 + b1*r_{t-1} + e_t
    if ("ar1" %in% baseline_models) {
      train_series <- train_df[[col]]
      ar1_fit <- lm(train_series[2:n_train_loc] ~ train_series[1:(n_train_loc - 1)])
      b0 <- coef(ar1_fit)[1]
      b1 <- coef(ar1_fit)[2]
      
      all_series <- c(train_df[[col]], val_df[[col]], test_df[[col]])
      forecast_indices <- (n_train_loc + 1):length(all_series)
      ar1_forecasts <- b0 + b1 * all_series[forecast_indices - 1]
      
      baseline_ar1_list[[col]] <- data.frame(
        Date = dates, Actual = actuals, Forecast = as.numeric(ar1_forecasts), Set = sets, Pair = col
      )
    }
  }

  # Save Baselines
  save_baseline <- function(name, results_list) {
    if (length(results_list) > 0) {
      dir_path <- file.path(results_dir, name)
      if (!dir.exists(dir_path)) dir.create(dir_path, recursive = TRUE)
      write_csv(do.call(rbind, results_list), file.path(dir_path, "forecasts.csv"))
      cat(sprintf("[SUCCESS] %s forecasts saved to %s\n", name, dir_path))
    }
  }

  save_baseline("baseline_rw", baseline_rw_list)
  save_baseline("baseline_mean", baseline_mean_list)
  save_baseline("baseline_ar1", baseline_ar1_list)
}

cat("\n--- Baseline Generation Completed. ---\n")
