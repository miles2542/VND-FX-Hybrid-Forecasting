import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import yaml
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load sensitive keys from .env
load_dotenv()


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def download_with_retry(ticker, start_date, end_date, retries=3):
    """Download with exponential backoff retry logic."""
    for i in range(retries):
        try:
            print(f"  [Attempt {i + 1}] Downloading {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                # Select 'Close' from MultiIndex or single Index
                if isinstance(data.columns, pd.MultiIndex):
                    if "Close" in data.columns.levels[0]:
                        col_data = data["Close"]
                    else:
                        return None
                else:
                    col = "Close" if "Close" in data.columns else "Adj Close"
                    col_data = data[col]

                res = (
                    col_data.iloc[:, 0]
                    if isinstance(col_data, pd.DataFrame)
                    else col_data
                )
                return res
        except Exception as e:
            wait = 2**i
            print(f"  [ERROR] {ticker} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return None


def check_local_cache(ticker_name, raw_path):
    """Check if raw file exists and is recent (within 24h)."""
    file_path = os.path.join(raw_path, f"{ticker_name}_raw.csv")
    if not os.path.exists(file_path):
        return None

    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    if datetime.now() - file_time < timedelta(hours=24):
        print(f"  [CACHE] Using local {ticker_name}_raw.csv.")
        return pd.read_csv(file_path, index_col=0, parse_dates=True).iloc[:, 0]
    return None


def download_and_cross_fx(config):
    """Build target-centric FX basket using a standardized USD-Pivot architecture."""
    target = config["active_target"]
    target_cfg = config["targets"][target]
    
    # Directional Map: True if ticker is CCY/USD (Base USD), False if USD/CCY (Term USD)
    # yfinance convention: EURUSD=X (Value of 1 EUR in USD) vs USDJPY=X (Value of 1 USD in JPY)
    # We want to standardize everything to USD_BASE: Price of 1 USD in CCY units.
    DIRECTION_MAP = {
        "EUR": ("EURUSD=X", True),
        "GBP": ("GBPUSD=X", True),
        "AUD": ("AUDUSD=X", True),
        "NZD": ("NZDUSD=X", True),
        "JPY": ("USDJPY=X", False),
        "CNY": ("USDCNY=X", False),
        "IDR": ("USDIDR=X", False),
        "THB": ("USDTHB=X", False),
        "PHP": ("USDPHP=X", False),
        "HKD": ("USDHKD=X", False),
        "SGD": ("USDSGD=X", False),
        "MYR": ("USDMYR=X", False),
        "VND": ("USDVND=X", False),
        "INR": ("USDINR=X", False),
        "CHF": ("USDCHF=X", False),
        "TRY": ("USDTRY=X", False),
        "KRW": ("USDKRW=X", False)
    }

    print(f"[INFO] Building {target}-centric FX basket using USD-Pivot...")
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    
    raw_path = os.path.join(config["paths"]["raw"], target)
    os.makedirs(raw_path, exist_ok=True)

    # 1. Collect all currencies involved
    all_ccys = {target} | set(target_cfg["partners"])
    if "USD" in all_ccys:
        all_ccys.add("USD")
    
    # 2. Extract USD_BASE for each (Price of 1 USD in CCY units)
    usd_pivot_rates = pd.DataFrame()
    
    for ccy in all_ccys:
        if ccy == "USD":
            usd_pivot_rates["USD"] = 1.0
            continue
            
        ticker, is_base_usd = DIRECTION_MAP.get(ccy, (f"USD{ccy}=X", False))
        
        # Download or Cache
        cached = check_local_cache(ccy, raw_path)
        if cached is not None:
            raw_data = cached
        else:
            raw_data = download_with_retry(ticker, start_date, end_date)
            if raw_data is not None:
                raw_data.to_csv(os.path.join(raw_path, f"{ccy}_raw.csv"))
            else:
                raise ValueError(f"[CRITICAL] Failed to download {ticker} for {ccy}!")

        # Normalize to USD_BASE: 1 USD = ? CCY
        if is_base_usd:
            usd_base = 1.0 / raw_data
        else:
            usd_base = raw_data
            
        if usd_pivot_rates.empty:
            usd_pivot_rates = pd.DataFrame(usd_base.rename(ccy))
        else:
            usd_pivot_rates = usd_pivot_rates.join(usd_base.rename(ccy), how="outer")

    # 3. Finalize Basket: Partner(A) / Target(B) = (USD/Target) / (USD/Partner)
    # This cancels out USD and gives us target units per partner unit.
    fx_final = pd.DataFrame(index=usd_pivot_rates.index)
    final_cols = []
    
    for partner in target_cfg["partners"]:
        pair_name = f"{partner}{target}"
        
        # Math: Price of 1 Partner Unit in Target Units
        # (USD/Target) / (USD/Partner) = (Target/USD) * (USD/Partner) = Target/Partner... No.
        # Logic: 1 USD = T Target Units. 1 USD = P Partner Units.
        # So P Partner Units = T Target Units.
        # 1 Partner Unit = T / P Target Units.
        fx_final[pair_name] = usd_pivot_rates[target] / usd_pivot_rates[partner]
        final_cols.append(pair_name)

    return fx_final[final_cols]


def download_interest_rates(config):
    """Download US interest rate as base proxy."""
    target = config["active_target"]
    raw_path = os.path.join(config["paths"]["raw"], target)
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]

    api_key = os.getenv("FRED_API_KEY")
    us_rate = pd.Series(dtype="float64")

    if api_key:
        cached = check_local_cache("us_interest_rate", raw_path)
        if cached is not None:
            us_rate = cached
        else:
            fred = Fred(api_key=api_key)
            for i in range(3):
                try:
                    us_rate = fred.get_series("DFF", observation_start=start_date, observation_end=end_date)
                    us_rate.to_csv(os.path.join(raw_path, "us_interest_rate_raw.csv"))
                    break
                except Exception as e:
                    time.sleep(2**i)
                    print(f"  [ERROR] FRED retry: {e}")

    # For now, local target rates are placeholders (handled by teammate later)
    target_rate = pd.Series(dtype="float64")
    return us_rate, target_rate


def preprocess_and_split(df_fx, us_rate, target_rate, config):
    """Clean, fill, transform, and split data into subfolders."""
    target = config["active_target"]
    print(f"[INFO] Preprocessing {target} basket...")

    last_market_date = pd.to_datetime(df_fx.index).max()
    df = df_fx.copy()
    df.index = pd.to_datetime(df.index)
    
    # Align Interest Rate
    df = df.join(us_rate.rename("REF_RATE"), how="outer")
    df["TARGET_RATE"] = 0.0 # Placeholder
    df = df[df.index <= last_market_date]

    # Robust Outlier Purge
    target_cfg = config["targets"][target]
    bounds = target_cfg.get("bounds", {})
    price_cols = df_fx.columns.tolist()

    for col in price_cols:
        # Step-median check (15% band fat-finger guard)
        local_median = df[col].rolling(window=5, center=True, min_periods=1).median()
        rel_mask = (df[col] / local_median < 0.93) | (df[col] / local_median > 1.07)
        
        # Scale-based guard
        b_min, b_max = bounds.get(col, (0, np.inf))
        abs_mask = (df[col] < b_min) | (df[col] > b_max)
        
        mask = rel_mask | abs_mask
        if mask.any():
            print(f"  [CLEAN] Purged {mask.sum()} outliers in {col}")
            df.loc[mask, col] = np.nan
            
    # DATA SANITY: 5-Day Business Week (Leakage & Autocorrelation Protocol)
    # 1. Drop Saturdays/Sundays to prevent zero-return weekend bias
    df = df[df.index.dayofweek < 5]

    # 2. Re-anchor to last REAL market date (No future-date padding)
    df = df[df.index <= last_market_date]

    # 3. Fill natural market gaps (holidays) via clean ffill logic
    df = df.ffill().bfill()

    # Log Returns
    ret_cols = []
    for col in price_cols:
        ret_name = f"{col}_RET"
        df[ret_name] = np.log(df[col] / df[col].shift(1)) * 100
        ret_cols.append(ret_name)

    df["IR_DIFF_LAGGED"] = (df["TARGET_RATE"] - df["REF_RATE"]).shift(1)
    df_clean = df.dropna(subset=ret_cols + ["IR_DIFF_LAGGED"]).copy()
    df_clean.index.name = "Date"

    # Split
    n = len(df_clean)
    s1 = int(n * config["data"]["train_ratio"])
    s2 = s1 + int(n * config["data"]["val_ratio"])
    
    train, val, test = df_clean.iloc[:s1], df_clean.iloc[s1:s2], df_clean.iloc[s2:]

    # Save to dynamic path
    proc_path = os.path.join(config["paths"]["processed"], target)
    os.makedirs(proc_path, exist_ok=True)
    
    df_clean.to_csv(os.path.join(proc_path, "fx_aligned.csv"))
    train.to_csv(os.path.join(proc_path, "train.csv"))
    val.to_csv(os.path.join(proc_path, "val.csv"))
    test.to_csv(os.path.join(proc_path, "test.csv"))

    print(f"[SUCCESS] {target} Preprocessed. Total: {len(df_clean)} | Path: {proc_path}")
    return df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    df_fx = download_and_cross_fx(cfg)
    us_rate, t_rate = download_interest_rates(cfg)
    preprocess_and_split(df_fx, us_rate, t_rate, cfg)
