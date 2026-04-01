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


def check_local_cache(ticker_name, raw_path, config_end_date):
    """Check if raw file exists and is recent (within 24h of target end)."""
    file_path = os.path.join(raw_path, f"{ticker_name}_raw.csv")
    if not os.path.exists(file_path):
        return None

    # Check modification time
    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

    # If file was updated recently or covers the end date, use it
    if datetime.now() - file_time < timedelta(hours=24):
        print(f"  [CACHE] Using local {ticker_name}_raw.csv (recent update).")
        return pd.read_csv(file_path, index_col=0, parse_dates=True).iloc[:, 0]
    return None


def download_and_cross_fx(config):
    """Fetch USD-VND and majors, then cross to get VND-centric pairs with Outer Merge."""
    print("[INFO] Building VND-centric FX basket via cross-rates...")
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    raw_path = config["paths"]["raw"]

    tickers_to_load = {
        "USDVND": "USDVND=X",
        "EURUSD": "EURUSD=X",
        "USDJPY": "USDJPY=X",
        "USDCNY": "USDCNY=X",
    }

    data_series = {}
    for name, ticker in tickers_to_load.items():
        # Check cache first
        cached = check_local_cache(name, raw_path, end_date)
        if cached is not None:
            data_series[name] = cached
        else:
            downloaded = download_with_retry(ticker, start_date, end_date)
            if downloaded is not None:
                downloaded.to_csv(os.path.join(raw_path, f"{name}_raw.csv"))
                data_series[name] = downloaded
            else:
                raise ValueError(
                    f"[CRITICAL] Could not download {ticker} after retries!"
                )

    # 1. Align on Outer Join to protect market outliers
    # Use full range from config to anchor the index
    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    fx_all = pd.DataFrame(index=idx)

    for name, s in data_series.items():
        fx_all = fx_all.join(s.rename(name), how="outer")

    # 2. Calculate Cross-Rates
    # EUR/VND = (EUR/USD) * (USD/VND)
    fx_all["EURVND"] = fx_all["EURUSD"] * fx_all["USDVND"]
    # JPY/VND = USD/VND / USD/JPY
    fx_all["JPYVND"] = fx_all["USDVND"] / fx_all["USDJPY"]
    # CNY/VND = USD/VND / USD/CNY
    fx_all["CNYVND"] = fx_all["USDVND"] / fx_all["USDCNY"]

    # 3. Clean and Save raw artifacts
    for pair in ["EURVND", "JPYVND", "CNYVND"]:
        fx_all[pair].to_csv(os.path.join(raw_path, f"{pair}_raw.csv"))

    return fx_all[["USDVND", "EURVND", "JPYVND", "CNYVND"]]


def download_interest_rates(config):
    """Download US interest rate from FRED with retry logic."""
    print("[INFO] Fetching interest rate data (FRED)...")
    raw_path = config["paths"]["raw"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]

    api_key = os.getenv("FRED_API_KEY")
    us_rate = pd.Series(dtype="float64")

    if api_key:
        # Check cache
        cached = check_local_cache("us_interest_rate", raw_path, end_date)
        if cached is not None:
            us_rate = cached
        else:
            fred = Fred(api_key=api_key)
            for i in range(3):
                try:
                    us_rate = fred.get_series(
                        "DFF", observation_start=start_date, observation_end=end_date
                    )
                    us_rate.to_csv(os.path.join(raw_path, "us_interest_rate_raw.csv"))
                    break
                except Exception as e:
                    time.sleep(2**i)
                    print(f"  [ERROR] FRED retry {i + 1}: {e}")

    vn_rate_path = os.path.join(raw_path, "vn_interest_rate_raw.csv")
    if os.path.exists(vn_rate_path):
        vn_rate = pd.read_csv(vn_rate_path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        print("[WARNING] Vietnam policy rate missing. creating 0 placeholder.")
        idx = pd.date_range(start=start_date, end=end_date, freq="D")
        vn_rate = pd.Series(0.0, index=idx)

    return us_rate, vn_rate


def preprocess_and_split(df_fx, us_rate, vn_rate, config):
    """Align fonts with Outer Join, apply Step-Function Ffill/Bfill, and split."""
    print("[INFO] Preprocessing with Outer-Join and Step-Function Alignment...")

    # Global Join
    df = df_fx.copy()
    df = df.join(us_rate.rename("US_RATE"), how="outer")
    df = df.join(vn_rate.rename("VN_RATE"), how="outer")

    # STEP-FUNCTION ALIGNMENT:
    # Use ffill() then bfill() to ensure regional holidays/weekends don't delete data
    df = df.ffill().bfill()

    # DATA SANITY: Robust Jump Detection (Median-based)
    # Defense: SBV manages VND within a ±5% band. A daily move > 15-20% is
    # statistically impossible for these currencies (USD, EUR, JPY, CNY)
    # and signifies a "fat finger" data error.
    price_cols = ["USDVND", "EURVND", "JPYVND", "CNYVND"]
    for col in price_cols:
        # 1. Compute a 5-day rolling median (TRAILING/NON-CENTERED) to provide a local baseline
        local_median = df[col].rolling(window=5, center=False, min_periods=1).median()

        # 2. Compare actual price to the local median
        ratio = df[col] / local_median

        # 3. Detect outliers (±15% deviation from past median)
        mask = (ratio < 0.85) | (ratio > 1.15)

        if mask.any():
            dates = df.index[mask].tolist()
            print(
                f"  [WARN] Data outlier detected in {col} on {dates}; imputing via ffill."
            )
            # Set outliers to NaN then ffill() from previously known clean data
            df.loc[mask, col] = np.nan
            df[col] = df[col].ffill()

    # Final cleanup for any edge-case NaNs
    df = df.ffill().bfill()

    # Calculate % log-returns (Lu & Perron 2010)
    ret_cols = []
    for col in ["USDVND", "EURVND", "JPYVND", "CNYVND"]:
        ret_name = f"{col}_RET"
        df[ret_name] = np.log(df[col] / df[col].shift(1)) * 100
        ret_cols.append(ret_name)

    # Interest Rate Differential (Exog) - Lagged t-1
    df["IR_DIFF"] = df["VN_RATE"] - df["US_RATE"]
    df["IR_DIFF_LAGGED"] = df["IR_DIFF"].shift(1)

    # Clean final set (only drop the first row with NaN return/lag)
    df_clean = df.dropna(subset=ret_cols + ["IR_DIFF_LAGGED"]).copy()
    df_clean.index.name = "Date"

    # Chronological Split
    n = len(df_clean)
    split1 = int(n * config["data"]["train_ratio"])
    split2 = split1 + int(n * config["data"]["val_ratio"])

    train = df_clean.iloc[:split1]
    val = df_clean.iloc[split1:split2]
    test = df_clean.iloc[split2:]

    # Save artifacts
    p = config["paths"]["processed"]
    df_clean.to_csv(os.path.join(p, "fx_aligned.csv"))
    train.to_csv(os.path.join(p, "train.csv"))
    val.to_csv(os.path.join(p, "val.csv"))
    test.to_csv(os.path.join(p, "test.csv"))

    print(
        f"[SUCCESS] Final Core Points: {len(df_clean)} | Train={len(train)}, Val={len(val)}, Test={len(test)}"
    )
    return df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    os.makedirs(cfg["paths"]["raw"], exist_ok=True)
    os.makedirs(cfg["paths"]["processed"], exist_ok=True)

    df_fx = download_and_cross_fx(cfg)
    us_rate, vn_rate = download_interest_rates(cfg)
    preprocess_and_split(df_fx, us_rate, vn_rate, cfg)
