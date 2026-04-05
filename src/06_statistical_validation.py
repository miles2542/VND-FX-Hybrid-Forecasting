import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, linear_reset
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
import statsmodels.api as sm


def set_reproducible_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_paths(config: dict) -> Dict[str, Path]:
    target = config["active_target"]
    return {
        "target": Path(target),
        "processed_dir": Path(config["paths"]["processed"]) / target,
        "results_dir": Path(config["paths"]["results"]) / target,
    }


def load_processed_frames(config: dict) -> Dict[str, pd.DataFrame]:
    paths = get_project_paths(config)
    out = {}
    for split in ["train", "val", "test", "fx_aligned"]:
        fpath = paths["processed_dir"] / f"{split}.csv"
        df = pd.read_csv(fpath)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        out[split] = df
    return out


def _read_forecast_file(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    forecast_col = "Hybrid" if "Hybrid" in df.columns else "Forecast"
    use_cols = ["Date", "Actual", forecast_col, "Set", "Pair"]
    clean = df[use_cols].copy()
    clean = clean.rename(columns={forecast_col: "Forecast"})
    clean["Model"] = model_name
    clean["Error"] = clean["Actual"] - clean["Forecast"]
    clean["SE"] = clean["Error"] ** 2
    clean["AE"] = clean["Error"].abs()
    return clean


def discover_forecasts(config: dict) -> pd.DataFrame:
    paths = get_project_paths(config)
    rows = []
    for model_dir in sorted(paths["results_dir"].iterdir()):
        if not model_dir.is_dir() or model_dir.name == "evaluation":
            continue
        fc_file = model_dir / "forecasts.csv"
        if fc_file.exists():
            rows.append(_read_forecast_file(fc_file, model_dir.name))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_arima_orders(config: dict) -> Dict[str, Tuple[int, int, int]]:
    paths = get_project_paths(config)
    fpath = paths["results_dir"] / "arima" / "diagnostics.json"
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {pair: tuple(v["order"]) for pair, v in data.items()}


def estimate_arima_param_significance(config: dict) -> pd.DataFrame:
    frames = load_processed_frames(config)
    train = frames["train"]
    orders = load_arima_orders(config)

    recs = []
    for pair, order in orders.items():
        y = train[pair].dropna()
        if y.empty:
            continue
        fit = ARIMA(y, order=order, trend="c").fit()
        table = pd.DataFrame(
            {
                "parameter": fit.params.index,
                "coef": fit.params.values,
                "std_err": fit.bse.values,
                "z_stat": fit.tvalues.values,
                "p_value": fit.pvalues.values,
                "conf_low": fit.conf_int().iloc[:, 0].values,
                "conf_high": fit.conf_int().iloc[:, 1].values,
            }
        )
        table["pair"] = pair
        table["order"] = str(order)
        recs.append(table)

    if not recs:
        return pd.DataFrame()
    out = pd.concat(recs, ignore_index=True)
    out["significant_5pct"] = out["p_value"] < 0.05
    return out[["pair", "order", "parameter", "coef", "std_err", "z_stat", "p_value", "conf_low", "conf_high", "significant_5pct"]]


def estimate_var_param_significance(config: dict) -> pd.DataFrame:
    frames = load_processed_frames(config)
    train = frames["train"]
    ret_cols = [c for c in train.columns if c.endswith("_RET")]

    paths = get_project_paths(config)
    diag_path = paths["results_dir"] / "var" / "diagnostics.json"
    selected_lag = 1
    if diag_path.exists():
        with open(diag_path, "r", encoding="utf-8") as f:
            selected_lag = int(json.load(f).get("selected_lag", 1))

    model = VAR(train[ret_cols].dropna())
    fit = model.fit(selected_lag, trend="ct")

    recs = []
    for eq_name in fit.params.columns:
        eq_table = pd.DataFrame(
            {
                "parameter": fit.params.index,
                "coef": fit.params[eq_name].values,
                "std_err": fit.stderr[eq_name].values,
                "t_stat": fit.tvalues[eq_name].values,
                "p_value": fit.pvalues[eq_name].values,
            }
        )
        eq_table["equation"] = eq_name
        recs.append(eq_table)

    out = pd.concat(recs, ignore_index=True)
    out["significant_5pct"] = out["p_value"] < 0.05
    return out[["equation", "parameter", "coef", "std_err", "t_stat", "p_value", "significant_5pct"]]


def diebold_mariano_test(
    actual: np.ndarray,
    forecast_a: np.ndarray,
    forecast_b: np.ndarray,
    h: int = 1,
    criterion: str = "mse",
) -> Dict[str, float]:
    y = np.asarray(actual)
    fa = np.asarray(forecast_a)
    fb = np.asarray(forecast_b)

    if not (len(y) == len(fa) == len(fb)):
        raise ValueError("Input arrays must have identical lengths.")
    if len(y) < 5:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": len(y)}

    ea = y - fa
    eb = y - fb

    if criterion == "mae":
        d = np.abs(ea) - np.abs(eb)
    else:
        d = ea**2 - eb**2

    n = len(d)
    d_bar = np.mean(d)

    # Newey-West style long-run variance for d_t (lag = h-1)
    gamma0 = np.var(d, ddof=1)
    if h > 1:
        gamma = []
        for lag in range(1, h):
            cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
            gamma.append(cov)
        long_run_var = gamma0 + 2.0 * np.sum(gamma)
    else:
        long_run_var = gamma0

    if long_run_var <= 0 or np.isnan(long_run_var):
        return {"dm_stat": np.nan, "p_value": np.nan, "n": n}

    dm_stat = d_bar / np.sqrt(long_run_var / n)

    # Harvey-Leybourne-Newbold small-sample correction
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_adj = dm_stat * hln
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_adj), df=n - 1))

    return {"dm_stat": float(dm_adj), "p_value": float(p_value), "n": int(n)}


def run_dm_comparisons(
    forecasts: pd.DataFrame,
    model_pairs: List[Tuple[str, str]],
    criterion: str = "mse",
    set_name: str = "test",
) -> pd.DataFrame:
    rows = []
    target_df = forecasts[forecasts["Set"] == set_name].copy()

    for pair in sorted(target_df["Pair"].unique()):
        pair_df = target_df[target_df["Pair"] == pair]
        for m1, m2 in model_pairs:
            df1 = pair_df[pair_df["Model"] == m1].sort_values("Date")
            df2 = pair_df[pair_df["Model"] == m2].sort_values("Date")
            if df1.empty or df2.empty:
                continue

            merged = df1[["Date", "Actual", "Forecast"]].merge(
                df2[["Date", "Forecast"]], on="Date", how="inner", suffixes=("_a", "_b")
            )
            if merged.empty:
                continue

            dm = diebold_mariano_test(
                actual=merged["Actual"].values,
                forecast_a=merged["Forecast_a"].values,
                forecast_b=merged["Forecast_b"].values,
                h=1,
                criterion=criterion,
            )
            rows.append(
                {
                    "Pair": pair,
                    "Model_A": m1,
                    "Model_B": m2,
                    "Loss": criterion,
                    "Set": set_name,
                    "DM_stat": dm["dm_stat"],
                    "p_value": dm["p_value"],
                    "n_obs": dm["n"],
                    "A_better_than_B_5pct": bool((dm["p_value"] < 0.05) and (dm["dm_stat"] < 0)),
                }
            )

    return pd.DataFrame(rows)


def residual_diagnostics_table(forecasts: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    rows = []
    for (pair, model, set_name), chunk in forecasts.groupby(["Pair", "Model", "Set"]):
        e = chunk["Error"].dropna().values
        if len(e) <= lags + 2:
            continue
        lb = acorr_ljungbox(e, lags=[lags], return_df=True)
        arch = het_arch(e, nlags=lags)
        jb_stat, jb_p = stats.jarque_bera(e)
        rows.append(
            {
                "Pair": pair,
                "Model": model,
                "Set": set_name,
                "N": len(e),
                "LB_pvalue": float(lb["lb_pvalue"].iloc[0]),
                "ARCHLM_pvalue": float(arch[1]),
                "JB_pvalue": float(jb_p),
                "LB_reject_5pct": float(lb["lb_pvalue"].iloc[0]) < 0.05,
                "ARCH_reject_5pct": float(arch[1]) < 0.05,
                "JB_reject_5pct": float(jb_p) < 0.05,
            }
        )

    return pd.DataFrame(rows)


def reset_specification_test(forecasts: pd.DataFrame, set_name: str = "test") -> pd.DataFrame:
    rows = []
    sub = forecasts[forecasts["Set"] == set_name].copy()

    for (pair, model), chunk in sub.groupby(["Pair", "Model"]):
        y = chunk["Actual"].values
        x = sm.add_constant(chunk["Forecast"].values)
        if len(y) < 20:
            continue
        ols = sm.OLS(y, x).fit()
        reset = linear_reset(ols, power=2, use_f=True)
        rows.append(
            {
                "Pair": pair,
                "Model": model,
                "Set": set_name,
                "R2": float(ols.rsquared),
                "RESET_F": float(reset.fvalue),
                "RESET_pvalue": float(reset.pvalue),
                "Specification_bias_5pct": float(reset.pvalue) < 0.05,
            }
        )

    return pd.DataFrame(rows)


def model_metrics_table(forecasts: pd.DataFrame, set_name: str = "test") -> pd.DataFrame:
    sub = forecasts[forecasts["Set"] == set_name].copy()
    grp = sub.groupby(["Pair", "Model"], as_index=False).agg(
        MSE=("SE", "mean"),
        MAE=("AE", "mean"),
        RMSE=("SE", lambda s: float(np.sqrt(np.mean(s)))),
    )
    return grp.sort_values(["Pair", "MSE"]).reset_index(drop=True)


def build_default_dm_pairs(models: List[str]) -> List[Tuple[str, str]]:
    comparisons = []

    # Hybrid vs base model
    base_map = {
        "hybrid_arima_svr": "arima",
        "hybrid_arima_mlp": "arima",
        "hybrid_var_svr": "var",
        "hybrid_var_mlp": "var",
    }
    for hybrid, base in base_map.items():
        if hybrid in models and base in models:
            comparisons.append((hybrid, base))

    # All models vs simple baselines
    baselines = [m for m in models if m.startswith("baseline_")]
    non_baselines = [m for m in models if not m.startswith("baseline_")]
    for m in non_baselines:
        for b in baselines:
            comparisons.append((m, b))

    return comparisons
