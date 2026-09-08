"""
Replace the flat 4.5% rate placeholder with a real daily SOFR rate series
(sourced from FRED — https://fred.stlouisfed.org/series/SOFR), and
recompute implied_vol using the real rate.

Before running:
  1. Download the SOFR series as CSV from:
     https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR&cosd=2025-09-01&coed=2026-06-30
  2. Save it to data/raw/sofr.csv

Run from the project root: python src/add_real_rate.py
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

FILES = {
    "train_prior_shock": "data/raw/brent_train_prior_shock.csv",
    "train_calm": "data/raw/brent_train_calm.csv",
    "test_current_shock": "data/raw/brent_test_current_shock.csv",
}

SOFR_PATH = "data/raw/sofr.csv"


def load_sofr():
    """
    FRED's CSV has columns typically named 'observation_date' and 'SOFR'
    (or similar) — the exact column names can vary slightly by download
    format, so this checks and adapts rather than assuming.
    """
    sofr = pd.read_csv(SOFR_PATH)
    print("SOFR file columns:", sofr.columns.tolist())

    date_col = [c for c in sofr.columns if "date" in c.lower()][0]
    rate_col = [c for c in sofr.columns if "sofr" in c.lower()][0]

    sofr = sofr.rename(columns={date_col: "trade_date", rate_col: "rate_pct"})
    sofr["trade_date"] = pd.to_datetime(sofr["trade_date"]).dt.date

    # FRED reports SOFR as a percentage (e.g. 4.50 meaning 4.50%) —
    # convert to decimal (0.045) for use in Black-76.
    sofr["rate"] = pd.to_numeric(sofr["rate_pct"], errors="coerce") / 100.0
    sofr = sofr[["trade_date", "rate"]].dropna()

    # SOFR is only published on business days. Build a full daily calendar
    # and forward-fill so every date in our option data has a rate,
    # including any weekend/holiday edge cases.
    full_range = pd.date_range(sofr["trade_date"].min(), sofr["trade_date"].max(), freq="D")
    sofr = sofr.set_index("trade_date").reindex(full_range.date).ffill().reset_index()
    sofr = sofr.rename(columns={"index": "trade_date"})
    return sofr


def black76_price(F, K, T, r, sigma, option_type="C"):
    if T <= 0 or sigma <= 0:
        return max(0.0, (F - K) if option_type == "C" else (K - F))
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if option_type == "C":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, r, option_type="C"):
    if price <= 0 or T <= 0 or pd.isna(F) or pd.isna(K) or pd.isna(r):
        return np.nan

    def objective(sigma):
        return black76_price(F, K, T, r, sigma, option_type) - price

    try:
        return brentq(objective, 1e-6, 5.0, xtol=1e-6)
    except ValueError:
        return np.nan


if __name__ == "__main__":
    sofr = load_sofr()
    print(f"Loaded SOFR: {len(sofr)} daily rates, "
          f"{sofr['rate'].min()*100:.2f}% to {sofr['rate'].max()*100:.2f}%")

    for window_name, path in FILES.items():
        print(f"\n--- {window_name} ---")
        df = pd.read_csv(path, parse_dates=["trade_date"])
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

        before_rate = df["rate"].mean() if "rate" in df.columns else None
        df = df.drop(columns=["rate"], errors="ignore").merge(sofr, on="trade_date", how="left")

        missing_rate = df["rate"].isna().sum()
        print(f"  rate missing after merge: {missing_rate} / {len(df)} rows")
        if before_rate is not None:
            print(f"  old flat rate was: {before_rate:.4f}, "
                  f"new mean real rate: {df['rate'].mean():.4f}")

        # Recompute implied_vol with the real rate
        df["implied_vol"] = df.apply(
            lambda row: implied_vol(
                price=row["settlement_price"],
                F=row["futures_settle"],
                K=row["strike_price"],
                T=row["ttm_years"],
                r=row["rate"],
                option_type=row["option_type"],
            ),
            axis=1,
        )
        print(f"  implied_vol populated for {df['implied_vol'].notna().sum()} / {len(df)} rows")

        out_path = path.replace(".csv", "_realrate.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")

    print("\nDone. Review the *_realrate.csv files, then replace the originals.")