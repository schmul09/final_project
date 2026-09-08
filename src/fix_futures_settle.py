"""
Fix for the futures_settle missing-data problem.

Root cause: the original extraction pulled futures prices from the
ohlcv-1d schema, which only produces a bar on days a contract actually
TRADED. Exchanges publish an official settlement price every day regardless
of trading activity (same reason the Statistics schema was used for the
OPTIONS settlement price) — so illiquid days for a given futures contract
month were silently missing a price even though one genuinely exists.

This script re-pulls ONLY the futures settlement prices (via Statistics,
stat_type=SETTLEMENT_PRICE — the same correct approach already used for
options), and re-merges them into the already-saved CSVs. It does NOT
re-pull the option chain data, which is unaffected by this bug and took
much longer to extract the first time.
"""

import os
import databento as db
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

API_KEY = os.environ.get("DATABENTO_API_KEY")
if not API_KEY:
    raise RuntimeError("Set DATABENTO_API_KEY before running this script.")

client = db.Historical(API_KEY)

DATASET = "GLBX.MDP3"
DATA_DIR = "/Users/dashazueva/dissertation_data"

FILES = {
    "train_prior_shock": (f"{DATA_DIR}/brent_train_prior_shock.csv", "2025-09-01", "2025-12-31"),
    "train_calm": (f"{DATA_DIR}/brent_train_calm.csv", "2026-01-01", "2026-02-15"),
    "test_current_shock": (f"{DATA_DIR}/brent_test_current_shock.csv", "2026-05-01", "2026-06-30"),
}

RATE_ASSUMPTION = 0.045


def _to_df_with_ts_column(data):
    df = data.to_df()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "index" in df.columns and "ts_event" not in df.columns:
            df = df.rename(columns={"index": "ts_event"})
    return df


def date_chunks(start, end, freq_days=14):
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    chunks = []
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + pd.Timedelta(days=freq_days), end_dt)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end
    return chunks


def get_futures_settlement_via_statistics(underlying_symbols, start, end):
    """
    Pull official settlement prices for the futures underlying, using the
    Statistics schema (correct — settlement price is published daily
    regardless of trading activity) instead of ohlcv-1d (incorrect — only
    exists on days with a trade).
    """
    frames = []
    for chunk_start, chunk_end in date_chunks(start, end):
        print(f"  ...pulling futures statistics {chunk_start} to {chunk_end}")
        try:
            data = client.timeseries.get_range(
                dataset=DATASET,
                schema="statistics",
                symbols=underlying_symbols,
                stype_in="raw_symbol",
                start=chunk_start,
                end=chunk_end,
            )
            df_chunk = _to_df_with_ts_column(data)
            if not df_chunk.empty:
                frames.append(df_chunk)
        except Exception as e:
            print(f"  WARNING: chunk {chunk_start}-{chunk_end} failed: {e}")
            continue
    if not frames:
        return pd.DataFrame()

    stats = pd.concat(frames, ignore_index=True)
    stats["trade_date"] = pd.to_datetime(stats["ts_event"]).dt.date
    stats["settlement_price"] = stats.loc[
        stats["stat_type"] == db.StatType.SETTLEMENT_PRICE, "price"
    ]
    settle = (
        stats.dropna(subset=["settlement_price"])
        .groupby(["trade_date", "symbol"], as_index=False)["settlement_price"]
        .last()
        .rename(columns={"symbol": "underlying", "settlement_price": "futures_settle_fixed"})
    )
    return settle


def implied_vol(price, F, K, T, r, option_type="C"):
    if price <= 0 or T <= 0 or pd.isna(F) or pd.isna(K):
        return np.nan

    def black76_price(F, K, T, r, sigma, option_type):
        if T <= 0 or sigma <= 0:
            return max(0.0, (F - K) if option_type == "C" else (K - F))
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        disc = np.exp(-r * T)
        if option_type == "C":
            return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    def objective(sigma):
        return black76_price(F, K, T, r, sigma, option_type) - price

    try:
        return brentq(objective, 1e-6, 5.0, xtol=1e-6)
    except ValueError:
        return np.nan


if __name__ == "__main__":
    for window_name, (path, start, end) in FILES.items():
        print(f"\n--- {window_name} ---")
        df = pd.read_csv(path, parse_dates=["trade_date", "expiration"])
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", dayfirst=True).dt.date

        underlying_symbols = df["underlying"].dropna().unique().tolist()
        print(f"  {len(underlying_symbols)} unique underlying contracts to fetch")

        settle = get_futures_settlement_via_statistics(underlying_symbols, start, end)
        if settle.empty:
            print("  WARNING: no futures settlement data returned — skipping this window.")
            continue

        before_missing = df["futures_settle"].isna().sum()
        df = df.drop(columns=["futures_settle"]).merge(
            settle, on=["trade_date", "underlying"], how="left"
        )
        df = df.rename(columns={"futures_settle_fixed": "futures_settle"})
        after_missing = df["futures_settle"].isna().sum()
        print(f"  futures_settle missing: {before_missing} -> {after_missing} "
              f"(out of {len(df)} rows)")

        # Recompute implied_vol now that futures_settle is fixed
        df["implied_vol"] = df.apply(
            lambda row: implied_vol(
                price=row["settlement_price"],
                F=row["futures_settle"],
                K=row["strike_price"],
                T=row["ttm_years"],
                r=RATE_ASSUMPTION,
                option_type=row["option_type"],
            ),
            axis=1,
        )
        print(f"  implied_vol now populated for {df['implied_vol'].notna().sum()} / {len(df)} rows")

        out_path = path.replace(".csv", "_fixed.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")

    print("\nDone. Review the *_fixed.csv files — if they look correct, replace "
          "the originals with these.")