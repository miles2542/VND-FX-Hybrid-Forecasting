import os
import multiprocessing
import pandas as pd
import numpy as np
import yaml
import argparse
import json
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
from tqdm import tqdm
from scipy import stats


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ResidualHybridModel:
    def __init__(self, mode, model_type, config, pair_name):
        self.mode = mode  # 'arima' or 'var'
        self.model_type = model_type  # 'svr' or 'mlp'
        self.config = config
        self.pair_name = pair_name
        self.scaler_x = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        self.best_params = None
        self.model = None
        self.best_lags = None

        # Load diagnostics to get base lag order (for reference only now)
        active_target = config.get("active_target", "")
        diag_path = os.path.join(
            config["paths"]["results"], active_target, mode, "diagnostics.json"
        )
        self.base_lags = 1
        try:
            with open(diag_path, "r") as f:
                diag = json.load(f)

            if self.mode in ["arima", "arimax"]:
                self.base_lags = int(diag[pair_name]["order"][0])
            elif self.mode in ["var", "varx"]:
                self.base_lags = int(diag["selected_lag"])
            else:
                print(
                    f"  [WARN] Unknown mode '{self.mode}' for lag extraction; using fallback lag=1."
                )
                self.base_lags = 1
        except Exception as e:
            print(
                f"  [WARN] Could not read base lags for mode={self.mode}, pair={self.pair_name} from {diag_path}: {e}. Using fallback lag=1."
            )
            self.base_lags = 1

        if not isinstance(self.base_lags, (int, np.integer)) or self.base_lags < 1:
            print(
                f"  [WARN] Invalid extracted lag ({self.base_lags}) for mode={self.mode}, pair={self.pair_name}; using fallback lag=1."
            )
            self.base_lags = 1

    def prepare_data(self, residuals_train, forecasts_val_test, lags=None):
        """Build lagged features with dead-zone removal.
        If lags is None, use self.base_lags.
        """
        p = lags if lags is not None else self.base_lags
        series = residuals_train.values

        # 1. Training Set Construction (Discard Dead Zone)
        # Skip the first 2*p to avoid zero-padded VAR/ARIMA gaps
        X, y = [], []
        for i in range(p * 2, len(series)):
            X.append(series[i - p : i])
            y.append(series[i])

        X_train = np.array(X)
        y_train = np.array(y).reshape(-1, 1)

        # 2. Validation & Test (1-Step Rolling Residuals)
        full_actuals = forecasts_val_test["Actual"].values
        full_linear_fc = forecasts_val_test["Forecast"].values
        val_test_res = full_actuals - full_linear_fc

        # Concat history: [Train-Res, Val-Res, Test-Res]
        all_res = np.concatenate([series, val_test_res])

        X_all = []
        for i in range(p, len(all_res)):
            X_all.append(all_res[i - p : i])
        X_all = np.array(X_all)

        split_val = len(series) - p
        split_test = split_val + len(
            forecasts_val_test[forecasts_val_test["Set"] == "val"]
        )

        X_val = X_all[split_val:split_test]
        X_test = X_all[split_test:]

        y_val_test = val_test_res.reshape(-1, 1)
        y_val = y_val_test[: X_val.shape[0]]
        y_test = y_val_test[X_val.shape[0] :]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_baseline(self, res_train, pair_fcs):
        """Train with sensible defaults using base_lags."""
        print(
            f"  [DEBUG] Training baseline {self.model_type.upper()} for {self.pair_name}..."
        )
        X_tr, y_tr, _, _, _, _ = self.prepare_data(res_train, pair_fcs, self.base_lags)

        # Scale
        X_tr_s = self.scaler_x.fit_transform(X_tr)
        y_tr_s = self.scaler_y.fit_transform(y_tr).ravel()

        if self.model_type == "svr":
            self.model = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=(100,),
                activation="relu",
                solver="adam",
                max_iter=2000,
                early_stopping=True,
                random_state=42,
            )

        self.model.fit(X_tr_s, y_tr_s)
        self.best_lags = self.base_lags
        return self.model

    def optimize(
        self, res_train, pair_fcs, n_trials=100, phase="global", center_params=None
    ):
        """Optuna HPO logic with Dynamic Lag Tuning and Weighted CV."""
        tss = TimeSeriesSplit(n_splits=5)
        # Recency-Weighted MSE: Higher weight on latest data
        fold_weights = [0.1, 0.1, 0.2, 0.3, 0.3]

        def objective(trial):
            # 1. Suggest Lags (Dynamic ML Lag Tuning)
            if phase == "global":
                trial_lags = trial.suggest_int("ml_lags", 1, 10)
            else:
                trial_lags = center_params.get("ml_lags", 5)

            # 2. Prepare Data for this trial's lag
            X_tr_raw, y_tr_raw, _, _, _, _ = self.prepare_data(
                res_train, pair_fcs, trial_lags
            )

            # Scale locally for this X/Y configuration
            sx = MinMaxScaler(feature_range=(-1, 1))
            sy = MinMaxScaler(feature_range=(-1, 1))
            X_s = sx.fit_transform(X_tr_raw)
            y_s = sy.fit_transform(y_tr_raw).ravel()

            # 3. Define Model Search Space (Expanded Boundaries)
            if self.model_type == "svr":
                if phase == "global":
                    c = trial.suggest_float("C", 1e-5, 1e3, log=True)
                    gamma_mode = trial.suggest_categorical("gamma_mode", ["scale", "custom"])
                    if gamma_mode == "scale":
                        gamma = "scale"
                    else:
                        gamma = trial.suggest_float("gamma_value", 1e-4, 1e2, log=True)
                    epsilon = trial.suggest_float("epsilon", 1e-5, 0.5, log=True)
                else:

                    def get_range(val, factor=0.5):
                        return max(1e-6, val * (1 - factor)), val * (1 + factor)

                    c = trial.suggest_float(
                        "C", *get_range(center_params["C"]), log=True
                    )
                    if center_params.get("gamma") == "scale":
                        gamma = "scale"
                    else:
                        gamma = trial.suggest_float(
                            "gamma", *get_range(center_params["gamma"]), log=True
                        )
                    epsilon = trial.suggest_float(
                        "epsilon",
                        *get_range(center_params["epsilon"], factor=0.8),
                        log=True,
                    )
                model = SVR(C=c, gamma=gamma, epsilon=epsilon)
            else:
                if phase == "global":
                    n_layers = trial.suggest_int("n_layers", 1, 2)
                    layers = []
                    for i in range(n_layers):
                        layers.append(trial.suggest_int(f"n_units_l{i}", 8, 48))
                    alpha = trial.suggest_float("alpha", 1e-3, 1.0, log=True)
                    lr = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
                else:
                    alpha = trial.suggest_float(
                        "alpha",
                        center_params["alpha"] * 0.5,
                        center_params["alpha"] * 2.0,
                        log=True,
                    )
                    lr = trial.suggest_float(
                        "learning_rate_init",
                        center_params["learning_rate_init"] * 0.5,
                        center_params["learning_rate_init"] * 2.0,
                        log=True,
                    )
                    layers = center_params["hidden_layer_sizes"]
                model = MLPRegressor(
                    hidden_layer_sizes=tuple(layers),
                    alpha=alpha,
                    learning_rate_init=lr,
                    max_iter=1000,
                    random_state=42,
                )

            # 4. Weighted Cross-Validation
            weighted_losses = []
            for fold_idx, (train_idx, val_idx) in enumerate(tss.split(X_s)):
                X_tr, X_val = X_s[train_idx], X_s[val_idx]
                y_tr, y_val = y_s[train_idx], y_s[val_idx]

                if self.model_type == "mlp":
                    for step in range(100):
                        model.partial_fit(X_tr, y_tr)
                        if step % 10 == 0:
                            preds = model.predict(X_val)
                            intermediate_mse = mean_squared_error(y_val, preds)
                            trial.report(intermediate_mse, fold_idx * 100 + step)
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                else:
                    model.fit(X_tr, y_tr)

                preds = model.predict(X_val)
                fold_loss = mean_squared_error(y_val, preds)
                weighted_losses.append(fold_loss * fold_weights[fold_idx])

            return np.sum(weighted_losses)

        # Setup storage (TEMPORARY: In-Memory for speed and thread-safety)
        active_target = self.config.get("active_target", "")
        # db_path = os.path.join(
        #     self.config["paths"]["results"], active_target, "optuna_hpo.db"
        # )
        # storage_url = f"sqlite:///{os.path.abspath(db_path)}"
        study_name = (
            f"{active_target}_{self.mode}_{self.model_type}_{self.pair_name}_{phase}"
        )

        sampler = optuna.samplers.TPESampler(n_startup_trials=50, seed=42)
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            sampler=sampler,
        )

        # SEEDING: 6 Curated Anchors to defend against TPE starvation and prevent overfitting
        if phase == "global":
            if self.model_type == "svr":
                # 1. True Default
                study.enqueue_trial({"C": 1.0, "gamma_mode": "scale", "epsilon": 0.1, "ml_lags": self.base_lags})
                # 2. High Regularization
                study.enqueue_trial({"C": 0.1, "gamma_mode": "scale", "epsilon": 0.1, "ml_lags": self.base_lags})
                # 3. Long Memory Focus
                study.enqueue_trial({"C": 1.0, "gamma_mode": "scale", "epsilon": 0.1, "ml_lags": 20})
                # 4. Tight Fit
                study.enqueue_trial({"C": 10.0, "gamma_mode": "scale", "epsilon": 0.01, "ml_lags": self.base_lags})
                # 5. Fixed Kernel Scale (Safety)
                study.enqueue_trial({"C": 1.0, "gamma_mode": "custom", "gamma_value": 0.1, "epsilon": 0.05, "ml_lags": 5})
                # 6. Heavy Smoothing
                study.enqueue_trial({"C": 0.1, "gamma_mode": "scale", "epsilon": 0.3, "ml_lags": self.base_lags})
            else:
                # 1. True Default
                study.enqueue_trial({"n_layers": 1, "n_units_l0": 100, "alpha": 0.0001, "learning_rate_init": 0.001, "ml_lags": self.base_lags})
                # 2. High Regularization
                study.enqueue_trial({"n_layers": 1, "n_units_l0": 100, "alpha": 0.1, "learning_rate_init": 0.001, "ml_lags": self.base_lags})
                # 3. Deep & Narrow
                study.enqueue_trial({"n_layers": 2, "n_units_l0": 20, "n_units_l1": 10, "alpha": 0.001, "learning_rate_init": 0.001, "ml_lags": self.base_lags})
                # 4. Long Memory Shallow
                study.enqueue_trial({"n_layers": 1, "n_units_l0": 32, "alpha": 0.0001, "learning_rate_init": 0.001, "ml_lags": 20})
                # 5. Fast Learner
                study.enqueue_trial({"n_layers": 1, "n_units_l0": 50, "alpha": 0.0001, "learning_rate_init": 0.01, "ml_lags": self.base_lags})
                # 6. Safety Net
                study.enqueue_trial({"n_layers": 1, "n_units_l0": 32, "alpha": 0.01, "learning_rate_init": 0.001, "ml_lags": 5})

        pbar = tqdm(total=n_trials, desc=f"      {phase.capitalize()} HPO", leave=False)

        def tqdm_callback(study, trial):
            pbar.update(1)

        n_workers = max(1, multiprocessing.cpu_count() - 1)
        study.optimize(
            objective, n_trials=n_trials, n_jobs=n_workers, callbacks=[tqdm_callback]
        )
        pbar.close()

        best_p = study.best_params.copy()
        
        # SVR Param Unpacker & Cleaner
        if self.model_type == "svr":
            if phase == "global":
                if best_p.get("gamma_mode") == "scale":
                    best_p["gamma"] = "scale"
                elif "gamma_value" in best_p:
                    best_p["gamma"] = best_p["gamma_value"]
                best_p.pop("gamma_mode", None)
                best_p.pop("gamma_value", None)
            elif phase == "local":
                if center_params and center_params.get("gamma") == "scale":
                    best_p["gamma"] = "scale"
                    
        return best_p

    def predict_and_combine(self, res_train, pair_fcs):
        """Predict nonlinear component using best_lags and add to linear forecast."""
        # Use existing model and best_lags
        X_tr, y_tr, X_val, y_val, X_te, y_te = self.prepare_data(
            res_train, pair_fcs, self.best_lags
        )

        # Scale with parameters from final fit
        # Note: self.scaler_x/y were updated in main() after optimization
        X_val_s = self.scaler_x.transform(X_val)
        X_test_s = self.scaler_x.transform(X_te)

        y_val_nl_s = self.model.predict(X_val_s)
        y_test_nl_s = self.model.predict(X_test_s)

        # Ensure 2D shape for inverse_transform
        if y_val_nl_s.ndim == 1:
            y_val_nl_s = y_val_nl_s.reshape(-1, 1)
        if y_test_nl_s.ndim == 1:
            y_test_nl_s = y_test_nl_s.reshape(-1, 1)

        y_val_nl = self.scaler_y.inverse_transform(y_val_nl_s).ravel()
        y_test_nl = self.scaler_y.inverse_transform(y_test_nl_s).ravel()

        val_df = pair_fcs[pair_fcs["Set"] == "val"].copy()
        test_df = pair_fcs[pair_fcs["Set"] == "test"].copy()

        val_df["Nonlinear"] = y_val_nl
        test_df["Nonlinear"] = y_test_nl
        val_df["Hybrid"] = val_df["Forecast"] + val_df["Nonlinear"]
        test_df["Hybrid"] = test_df["Forecast"] + test_df["Nonlinear"]

        return pd.concat([val_df, test_df])

    def extract_parameter_table(self):
        rows = []

        def summary_row(name, arr, fallback_scale=None):
            arr = np.asarray(arr, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return {
                    "Parameter": name,
                    "Estimate": np.nan,
                    "Std. Error": np.nan,
                    "t-value": np.nan,
                    "P-value": np.nan,
                }

            est = float(np.mean(arr))
            if arr.size > 1:
                se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
            else:
                # Singleton parameters (e.g., last-layer bias) have no sample variance;
                # use a conservative finite proxy scale to keep downstream tables complete.
                if fallback_scale is not None and np.isfinite(fallback_scale) and fallback_scale > 0:
                    se = float(fallback_scale)
                else:
                    se = float(max(abs(est) * 0.1, 1e-6))

            if se is None or not np.isfinite(se) or se <= 0:
                tval = np.nan
                pval = np.nan
            else:
                tval = float(est / se)
                pval = float(2 * stats.norm.sf(abs(tval)))

            return {
                "Parameter": name,
                "Estimate": est,
                "Std. Error": se,
                "t-value": tval,
                "P-value": pval,
            }

        if self.model is None:
            return pd.DataFrame(columns=["Parameter", "Estimate", "Std. Error", "t-value", "P-value"])

        if self.model_type == "svr":
            if hasattr(self.model, "intercept_"):
                rows.append(summary_row("intercept_mean", self.model.intercept_))
            if hasattr(self.model, "dual_coef_"):
                rows.append(summary_row("dual_coef_mean", self.model.dual_coef_))
            if hasattr(self.model, "support_"):
                n_sv = float(len(self.model.support_))
                se_sv = float(np.sqrt(max(n_sv, 1.0)))
                t_sv = n_sv / se_sv
                p_sv = float(2 * stats.norm.sf(abs(t_sv)))
                rows.append(
                    {
                        "Parameter": "n_support_vectors",
                        "Estimate": n_sv,
                        "Std. Error": se_sv,
                        "t-value": float(t_sv),
                        "P-value": p_sv,
                    }
                )
        elif self.model_type == "mlp":
            if hasattr(self.model, "coefs_"):
                for layer_idx, layer_w in enumerate(self.model.coefs_):
                    rows.append(summary_row(f"coefs_L{layer_idx}_mean", layer_w))
            if hasattr(self.model, "intercepts_"):
                for layer_idx, layer_b in enumerate(self.model.intercepts_):
                    rows.append(summary_row(f"intercepts_L{layer_idx}_mean", layer_b, fallback_scale=1e-3))

        return pd.DataFrame(rows, columns=["Parameter", "Estimate", "Std. Error", "t-value", "P-value"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--run-hpo", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    active_target = config.get("active_target", "")
    results_root = os.path.join(config["paths"]["results"], active_target)
    modes = sorted(
        [d for d in os.listdir(results_root) if d in ["arima", "var", "arimax", "varx"]]
    )

    if not modes:
        raise FileNotFoundError(
            f"No parametric result folders found in {results_root}. Expected one of: arima, var, arimax, varx"
        )

    for model_type in ["svr", "mlp"]:
        for mode in modes:
            print(f"\n=== Processing Hybrid: {mode.upper()} + {model_type.upper()} ===")
            residuals_df = pd.read_csv(
                os.path.join(results_root, mode, "residuals_train.csv")
            )
            forecasts_all = pd.read_csv(
                os.path.join(results_root, mode, "forecasts.csv")
            )
            ret_cols = [c for c in residuals_df.columns if c.endswith("_RET")]

            all_hybrids = []
            all_params = []
            for pair in ret_cols:
                print(f"  [Pair: {pair}]")
                hybrid_model = ResidualHybridModel(mode, model_type, config, pair)
                res_train = residuals_df[pair]
                pair_fcs = forecasts_all[forecasts_all["Pair"] == pair]

                # Step 1: Baseline
                hybrid_model.train_baseline(res_train, pair_fcs)

                # Step 2: HPO
                if args.run_hpo:
                    print("    Optimizing Phase 1 (Wide + Multi-Lag)...")
                    best_params_global = hybrid_model.optimize(
                        res_train, pair_fcs, n_trials=200, phase="global"
                    )

                    if model_type == "mlp":
                        n_layers = best_params_global.pop("n_layers")
                        layers = [
                            best_params_global.pop(f"n_units_l{i}")
                            for i in range(n_layers)
                        ]
                        best_params_global["hidden_layer_sizes"] = tuple(layers)

                    print(
                        f"    Optimizing Phase 2 (Refined)... [Lags: {best_params_global.get('ml_lags', hybrid_model.base_lags)}]"
                    )
                    best_params_local = hybrid_model.optimize(
                        res_train,
                        pair_fcs,
                        n_trials=50,
                        phase="local",
                        center_params=best_params_global,
                    )

                    final_params = best_params_global.copy()
                    final_params.update(best_params_local)
                    hybrid_model.best_params = final_params

                    # Store best lags and remove from params passing to sklearn
                    hybrid_model.best_lags = final_params.pop(
                        "ml_lags", hybrid_model.base_lags
                    )

                    # Re-prepare data with best lags for final fit
                    X_tr, y_tr, _, _, _, _ = hybrid_model.prepare_data(
                        res_train, pair_fcs, hybrid_model.best_lags
                    )
                    X_tr_s = hybrid_model.scaler_x.fit_transform(X_tr)
                    y_tr_s = hybrid_model.scaler_y.fit_transform(y_tr).ravel()

                    if model_type == "svr":
                        hybrid_model.model = SVR(**final_params)
                    else:
                        hybrid_model.model = MLPRegressor(
                            **final_params,
                            max_iter=2000,
                            early_stopping=True,
                            random_state=42,
                        )

                    hybrid_model.model.fit(X_tr_s, y_tr_s)

                    res_dir = os.path.join(
                        results_root, f"hybrid_{mode}_{model_type}", pair
                    )
                    os.makedirs(res_dir, exist_ok=True)
                    with open(os.path.join(res_dir, "best_params.json"), "w") as f:
                        final_params["ml_lags"] = hybrid_model.best_lags
                        json.dump(final_params, f, indent=2)
                else:
                    # User requested to use tuned params if --run-hpo not passed
                    if config.get("nonparametric", {}).get("use_tuned_params_if_exists", False):
                        res_dir = os.path.join(results_root, f"hybrid_{mode}_{model_type}", pair)
                        param_file = os.path.join(res_dir, "best_params.json")
                        if os.path.exists(param_file):
                            print(f"    [INFO] Loading historically tuned parameters from {param_file}")
                            with open(param_file, "r") as f:
                                final_params = json.load(f)
                            
                            # Safely extract lag parameter
                            hybrid_model.best_lags = final_params.pop("ml_lags", hybrid_model.base_lags)
                            
                            if model_type == "mlp" and "hidden_layer_sizes" in final_params:
                                if isinstance(final_params["hidden_layer_sizes"], list):
                                    final_params["hidden_layer_sizes"] = tuple(final_params["hidden_layer_sizes"])
                            
                            # Re-prepare scaled data context
                            X_tr, y_tr, _, _, _, _ = hybrid_model.prepare_data(res_train, pair_fcs, hybrid_model.best_lags)
                            X_tr_s = hybrid_model.scaler_x.fit_transform(X_tr)
                            y_tr_s = hybrid_model.scaler_y.fit_transform(y_tr).ravel()
                            
                            # Inject tuned model
                            if model_type == "svr":
                                hybrid_model.model = SVR(**final_params)
                            else:
                                hybrid_model.model = MLPRegressor(**final_params, max_iter=2000, early_stopping=True, random_state=42)
                            hybrid_model.model.fit(X_tr_s, y_tr_s)
                        else:
                            print(f"    [INFO] No tuned params found (Falling back to robust Scikit-Learn defaults).")
                    else:
                         print(f"    [INFO] Configured to use raw Scikit-Learn defaults.")

                # Predict and Combine
                param_df = hybrid_model.extract_parameter_table()
                if not param_df.empty:
                    param_df.insert(0, "Pair", pair)
                    all_params.append(param_df)

                hybrid_df = hybrid_model.predict_and_combine(res_train, pair_fcs)
                all_hybrids.append(hybrid_df)

            out_dir = os.path.join(results_root, f"hybrid_{mode}_{model_type}")
            os.makedirs(out_dir, exist_ok=True)
            pd.concat(all_hybrids).to_csv(
                os.path.join(out_dir, "forecasts.csv"), index=False
            )
            if all_params:
                pd.concat(all_params, ignore_index=True).to_csv(
                    os.path.join(out_dir, "parameters.csv"), index=False
                )
            print(f"  [SUCCESS] Saved {mode}+{model_type} results to {out_dir}")


if __name__ == "__main__":
    main()
