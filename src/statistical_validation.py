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
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

try:
    from statsmodels.tsa.stattools import phillips_perron
except Exception:
    phillips_perron = None

try:
    import ruptures as rpt
except Exception:
    rpt = None


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
        if np.nanstd(chunk["Forecast"].values) == 0 or np.unique(chunk["Forecast"].dropna().values).size < 2:
            rows.append(
                {
                    "Pair": pair,
                    "Model": model,
                    "Set": set_name,
                    "R2": np.nan,
                    "RESET_F": np.nan,
                    "RESET_pvalue": np.nan,
                    "Specification_bias_5pct": False,
                    "RESET_note": "Skipped: forecast is constant, RESET not defined.",
                }
            )
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
                "RESET_note": "OK",
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


def rolling_parameter_stability(config: dict, split: str = "train", window: int = 252, step: int = 20) -> pd.DataFrame:
    frames = load_processed_frames(config)
    df = frames[split].copy()
    ret_cols = [c for c in df.columns if c.endswith("_RET")]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    rows = []
    for pair in ret_cols:
        series = df[pair].dropna()
        if len(series) < window + 10:
            continue
        dates = series.index if isinstance(series.index, pd.DatetimeIndex) else pd.RangeIndex(len(series))
        for end_idx in range(window, len(series) + 1, max(1, step)):
            window_series = series.iloc[end_idx - window:end_idx]
            try:
                fit = ARIMA(window_series, order=(1, 0, 1), trend="c").fit()
                rows.append(
                    {
                        "Pair": pair,
                        "Date": dates[end_idx - 1],
                        "ar1": float(fit.params.get("ar.L1", np.nan)),
                        "ma1": float(fit.params.get("ma.L1", np.nan)),
                        "const": float(fit.params.get("const", np.nan)),
                        "aic": float(fit.aic),
                    }
                )
            except Exception:
                continue

    return pd.DataFrame(rows)


def garch_residual_diagnostics(forecasts: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    rows = []
    try:
        from arch import arch_model
    except Exception:
        return pd.DataFrame(rows)

    for (pair, model), chunk in forecasts.groupby(["Pair", "Model"]):
        resid = chunk["Error"].dropna().values * 100
        if len(resid) < 50:
            continue
        try:
            fit = arch_model(resid, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
            rows.append(
                {
                    "Pair": pair,
                    "Model": model,
                    "omega": float(fit.params.get("omega", np.nan)),
                    "alpha1": float(fit.params.get("alpha[1]", np.nan)),
                    "beta1": float(fit.params.get("beta[1]", np.nan)),
                    "persistence": float(fit.params.get("alpha[1]", np.nan) + fit.params.get("beta[1]", np.nan)),
                    "aic": float(fit.aic),
                    "loglik": float(fit.loglikelihood),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def var_irf_table(config: dict, split: str = "train", steps: int = 10) -> pd.DataFrame:
    frames = load_processed_frames(config)
    train = frames[split].copy()
    ret_cols = [c for c in train.columns if c.endswith("_RET")]
    if len(ret_cols) < 2:
        return pd.DataFrame()

    paths = get_project_paths(config)
    diag_path = paths["results_dir"] / "var" / "diagnostics.json"
    lag = 1
    if diag_path.exists():
        with open(diag_path, "r", encoding="utf-8") as f:
            lag = int(json.load(f).get("selected_lag", 1))

    fit = VAR(train[ret_cols].dropna()).fit(lag)
    irf = fit.irf(steps)
    records = []
    for response_idx, response in enumerate(ret_cols):
        for shock_idx, shock in enumerate(ret_cols):
            arr = irf.irfs[:, response_idx, shock_idx]
            for step, value in enumerate(arr):
                records.append({"response": response, "shock": shock, "step": step, "irf": float(value)})
    return pd.DataFrame(records)


def event_annotation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Date": "2022-05-05", "Event": "BSP rate hike cycle intensifies", "Category": "policy"},
            {"Date": "2023-02-01", "Event": "Fed tightening / USD strength repricing", "Category": "global risk"},
            {"Date": "2023-11-01", "Event": "China growth and trade surprise", "Category": "trade"},
            {"Date": "2024-08-01", "Event": "Risk-off shock / VIX spike template", "Category": "risk"},
            {"Date": "2025-03-01", "Event": "Remittance and balance-of-payments support", "Category": "external flow"},
        ]
    )


def get_return_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("_RET")]


def get_returns_frame(config: dict, split: str = "fx_aligned") -> pd.DataFrame:
    frames = load_processed_frames(config)
    df = frames[split].copy()
    ret_cols = get_return_columns(df)
    out = df[["Date"] + ret_cols].copy() if "Date" in df.columns else df[ret_cols].copy()
    if "Date" in out.columns:
        out = out.dropna().set_index("Date")
    return out


def stationarity_tests_for_series(series: pd.Series, regression: str = "c") -> Dict[str, float]:
    s = series.dropna().astype(float)
    if len(s) < 30:
        return {
            "n": len(s),
            "adf_stat": np.nan,
            "adf_pvalue": np.nan,
            "kpss_stat": np.nan,
            "kpss_pvalue": np.nan,
            "pp_stat": np.nan,
            "pp_pvalue": np.nan,
        }

    adf = adfuller(s, regression=regression, autolag="AIC")
    kpss_out = kpss(s, regression=("ct" if regression == "ct" else "c"), nlags="auto")
    if phillips_perron is not None:
        pp = phillips_perron(s, trend=("ct" if regression == "ct" else "c"))
        pp_stat = float(pp[0])
        pp_p = float(pp[1])
    else:
        pp_stat, pp_p = np.nan, np.nan

    return {
        "n": len(s),
        "adf_stat": float(adf[0]),
        "adf_pvalue": float(adf[1]),
        "kpss_stat": float(kpss_out[0]),
        "kpss_pvalue": float(kpss_out[1]),
        "pp_stat": pp_stat,
        "pp_pvalue": pp_p,
    }


def run_stationarity_panel(config: dict, split: str = "fx_aligned") -> pd.DataFrame:
    df = get_returns_frame(config, split=split)
    rows = []
    for col in df.columns:
        c_only = stationarity_tests_for_series(df[col], regression="c")
        c_only.update({"Pair": col, "Regression": "constant"})
        rows.append(c_only)

        c_trend = stationarity_tests_for_series(df[col], regression="ct")
        c_trend.update({"Pair": col, "Regression": "constant+trend"})
        rows.append(c_trend)

    out = pd.DataFrame(rows)
    out["ADF_reject_5pct"] = out["adf_pvalue"] < 0.05
    out["KPSS_reject_5pct"] = out["kpss_pvalue"] < 0.05
    out["PP_reject_5pct"] = out["pp_pvalue"] < 0.05
    return out


def _sup_chow_mean_break(series: pd.Series, min_frac: float = 0.15) -> Dict[str, float]:
    s = series.dropna().astype(float)
    n = len(s)
    if n < 80:
        return {"break_index": np.nan, "break_date": pd.NaT, "supF": np.nan, "pvalue": np.nan}

    lo = int(n * min_frac)
    hi = int(n * (1 - min_frac))
    full_mean = s.mean()
    sse_pool = np.sum((s - full_mean) ** 2)

    best_f = -np.inf
    best_i = None
    for i in range(lo, hi):
        s1 = s.iloc[:i]
        s2 = s.iloc[i:]
        sse_1 = np.sum((s1 - s1.mean()) ** 2)
        sse_2 = np.sum((s2 - s2.mean()) ** 2)
        k = 1
        denom = (sse_1 + sse_2) / max(n - 2 * k, 1)
        if denom <= 0:
            continue
        f_val = ((sse_pool - (sse_1 + sse_2)) / k) / denom
        if f_val > best_f:
            best_f = f_val
            best_i = i

    if best_i is None:
        return {"break_index": np.nan, "break_date": pd.NaT, "supF": np.nan, "pvalue": np.nan}

    pval = 1 - stats.f.cdf(best_f, 1, max(n - 2, 1))
    return {
        "break_index": int(best_i),
        "break_date": s.index[best_i] if hasattr(s.index, "dtype") else pd.NaT,
        "supF": float(best_f),
        "pvalue": float(pval),
    }


def structural_breaks_panel(config: dict, split: str = "fx_aligned", max_breaks: int = 3) -> pd.DataFrame:
    df = get_returns_frame(config, split=split)
    rows = []

    for col in df.columns:
        s = df[col].dropna()
        if rpt is not None and len(s) >= 80:
            signal = s.values.reshape(-1, 1)
            algo = rpt.Binseg(model="l2").fit(signal)
            bkps = algo.predict(n_bkps=max_breaks)
            for idx, b in enumerate(bkps[:-1], start=1):
                rows.append(
                    {
                        "Pair": col,
                        "method": "ruptures_binseg",
                        "break_number": idx,
                        "break_index": int(b),
                        "break_date": s.index[min(b - 1, len(s) - 1)],
                        "score": np.nan,
                        "pvalue": np.nan,
                    }
                )
        else:
            chow = _sup_chow_mean_break(s)
            rows.append(
                {
                    "Pair": col,
                    "method": "sup_chow_mean",
                    "break_number": 1,
                    "break_index": chow["break_index"],
                    "break_date": chow["break_date"],
                    "score": chow["supF"],
                    "pvalue": chow["pvalue"],
                }
            )

    return pd.DataFrame(rows)


def johansen_cointegration_table(config: dict, split: str = "train", det_order: int = 0, k_ar_diff: int = 1) -> pd.DataFrame:
    df = get_returns_frame(config, split=split)
    if df.shape[1] < 2:
        return pd.DataFrame()
    joh = coint_johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)
    out = pd.DataFrame(
        {
            "rank_r": np.arange(len(joh.lr1)),
            "trace_stat": joh.lr1,
            "crit_90": joh.cvt[:, 0],
            "crit_95": joh.cvt[:, 1],
            "crit_99": joh.cvt[:, 2],
        }
    )
    out["reject_95pct"] = out["trace_stat"] > out["crit_95"]
    return out


def select_var_lag_aic(config: dict, split: str = "train", maxlags: int = 10) -> Dict[str, float]:
    df = get_returns_frame(config, split=split)
    model = VAR(df)
    sel = model.select_order(maxlags=maxlags)
    return {
        "aic_lag": int(sel.aic) if sel.aic is not None else np.nan,
        "bic_lag": int(sel.bic) if sel.bic is not None else np.nan,
        "hqic_lag": int(sel.hqic) if sel.hqic is not None else np.nan,
    }


def granger_causality_matrix(config: dict, split: str = "train", maxlag: int = 5) -> pd.DataFrame:
    df = get_returns_frame(config, split=split)
    cols = list(df.columns)
    out = pd.DataFrame(np.nan, index=cols, columns=cols)

    for caused in cols:
        for causing in cols:
            if caused == causing:
                out.loc[caused, causing] = np.nan
                continue
            pair_df = df[[caused, causing]].dropna()
            if len(pair_df) < (maxlag + 10):
                continue
            try:
                res = grangercausalitytests(pair_df[[caused, causing]], maxlag=maxlag, verbose=False)
                pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)]
                out.loc[caused, causing] = float(np.min(pvals))
            except Exception:
                out.loc[caused, causing] = np.nan
    return out


def acf_pacf_sqacf_table(config: dict, split: str = "fx_aligned", nlags: int = 40) -> pd.DataFrame:
    from statsmodels.tsa.stattools import acf, pacf

    df = get_returns_frame(config, split=split)
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) <= nlags + 5:
            continue
        acf_vals = acf(s, nlags=nlags, fft=True)
        pacf_vals = pacf(s, nlags=nlags)
        sq_acf_vals = acf(s**2, nlags=nlags, fft=True)
        ci = 1.96 / np.sqrt(len(s))
        for lag in range(1, nlags + 1):
            rows.append(
                {
                    "Pair": col,
                    "Lag": lag,
                    "ACF": float(acf_vals[lag]),
                    "PACF": float(pacf_vals[lag]),
                    "SqACF": float(sq_acf_vals[lag]),
                    "sig_bound": float(ci),
                }
            )
    return pd.DataFrame(rows)


def rolling_diagnostics(config: dict, split: str = "fx_aligned", window: int = 252, arch_lags: int = 5) -> pd.DataFrame:
    df = get_returns_frame(config, split=split)
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < window + 5:
            continue
        for i in range(window, len(s) + 1):
            chunk = s.iloc[i - window:i]
            lb = acorr_ljungbox(chunk, lags=[1], return_df=True)
            arch = het_arch(chunk.values, nlags=arch_lags)
            rows.append(
                {
                    "Date": chunk.index[-1],
                    "Pair": col,
                    "rolling_vol_252": float(chunk.std()),
                    "rolling_autocorr_l1": float(chunk.autocorr(lag=1)),
                    "rolling_lb_pvalue_l1": float(lb["lb_pvalue"].iloc[0]),
                    "rolling_arch_pvalue": float(arch[1]),
                }
            )
    return pd.DataFrame(rows)


def cross_correlation_lead_lag(config: dict, split: str = "fx_aligned", max_lag: int = 20) -> pd.DataFrame:
    df = get_returns_frame(config, split=split)
    cols = list(df.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            x = df[a].dropna()
            y = df[b].dropna()
            aligned = pd.concat([x, y], axis=1).dropna()
            if aligned.empty:
                continue
            x = aligned[a]
            y = aligned[b]
            best_lag = 0
            best_corr = 0.0
            for lag in range(-max_lag, max_lag + 1):
                corr = x.corr(y.shift(lag))
                if pd.notna(corr) and abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            rows.append(
                {
                    "Pair_A": a,
                    "Pair_B": b,
                    "best_lag": int(best_lag),
                    "best_corr": float(best_corr),
                    "interpretation": "A_leads_B" if best_lag > 0 else ("B_leads_A" if best_lag < 0 else "synchronous"),
                }
            )
    return pd.DataFrame(rows)


def rolling_correlation_table(config: dict, split: str = "fx_aligned", window: int = 126) -> pd.DataFrame:
    df = get_returns_frame(config, split=split)
    cols = list(df.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pair = df[[a, b]].dropna()
            if len(pair) < window + 5:
                continue
            corr_s = pair[a].rolling(window).corr(pair[b]).dropna()
            for dt, val in corr_s.items():
                rows.append({"Date": dt, "Pair_A": a, "Pair_B": b, "rolling_corr": float(val)})
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def rolling_parameter_stability(config: dict, split: str = "train", window: int = 252) -> pd.DataFrame:
    frames = load_processed_frames(config)
    df = frames[split].copy()
    ret_cols = [c for c in df.columns if c.endswith("_RET")]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    rows = []
    for pair in ret_cols:
        series = df[pair].dropna()
        if len(series) < window + 10:
            continue
        dates = series.index if isinstance(series.index, pd.DatetimeIndex) else pd.RangeIndex(len(series))
        for end_idx in range(window, len(series) + 1):
            window_series = series.iloc[end_idx - window:end_idx]
            try:
                fit = ARIMA(window_series, order=(1, 0, 1), trend="c").fit()
                rows.append(
                    {
                        "Pair": pair,
                        "Date": dates[end_idx - 1],
                        "ar1": float(fit.params.get("ar.L1", np.nan)),
                        "ma1": float(fit.params.get("ma.L1", np.nan)),
                        "const": float(fit.params.get("const", np.nan)),
                        "aic": float(fit.aic),
                    }
                )
            except Exception:
                continue
    return pd.DataFrame(rows)


def var_irf_table(config: dict, split: str = "train", steps: int = 10) -> pd.DataFrame:
    frames = load_processed_frames(config)
    train = frames[split].copy()
    ret_cols = [c for c in train.columns if c.endswith("_RET")]
    if len(ret_cols) < 2:
        return pd.DataFrame()

    paths = get_project_paths(config)
    diag_path = paths["results_dir"] / "var" / "diagnostics.json"
    lag = 1
    if diag_path.exists():
        with open(diag_path, "r", encoding="utf-8") as f:
            lag = int(json.load(f).get("selected_lag", 1))

    fit = VAR(train[ret_cols].dropna()).fit(lag)
    irf = fit.irf(steps)
    records = []
    for response_idx, response in enumerate(ret_cols):
        for shock_idx, shock in enumerate(ret_cols):
            arr = irf.irfs[:, response_idx, shock_idx]
            for step, value in enumerate(arr):
                records.append({"response": response, "shock": shock, "step": step, "irf": float(value)})
    return pd.DataFrame(records)


def event_annotation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Date": "2022-05-05", "Event": "BSP rate hike cycle intensifies", "Category": "policy"},
            {"Date": "2023-02-01", "Event": "Fed tightening / USD strength repricing", "Category": "global risk"},
            {"Date": "2023-11-01", "Event": "China growth and trade surprise", "Category": "trade"},
            {"Date": "2024-08-01", "Event": "Risk-off shock / VIX spike template", "Category": "risk"},
            {"Date": "2025-03-01", "Event": "Remittance and balance-of-payments support", "Category": "external flow"},
        ]
    )
