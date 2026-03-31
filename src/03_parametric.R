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
processed_dir <- config$paths$processed
results_dir <- config$paths$results

# 1. Load Data
train_df <- read_csv(file.path(processed_dir, "train.csv"), show_col_types = FALSE)
val_df <- read_csv(file.path(processed_dir, "val.csv"), show_col_types = FALSE)
test_df <- read_csv(file.path(processed_dir, "test.csv"), show_col_types = FALSE)

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

cat("\n--- Parametric Stage Completed. Results saved to results/arima/ and results/var/ ---\n")
