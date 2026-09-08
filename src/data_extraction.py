"""
Databento data extraction script for MSc dissertation:
Neural network fair-value estimation of energy derivative options,
with a hedging application to Russian refined-product shortage risk.

CONFIRMED so far (from live catalog checks):
  - Brent EU-style vanilla option: ticker root "BE" ("Brent Last Day Financial
    European Option"), dataset GLBX.MDP3 (CME Globex), underlying futures
    root "BZ". Historical coverage since 2010-11-14. Parent/smart symbology
    (BE.OPT) resolves correctly for this product.
  - Gasoil EU-style vanilla option: ticker root "F8" ("European-Style Low
    Sulphur Gasoil Option"), ALSO on dataset GLBX.MDP3 (CME Globex). Historical
    coverage since 2022-06-28. IMPORTANT: neither "F8" nor "BG" resolve as
    parent/smart symbols on Databento (confirmed via direct diagnostic testing
    — both F8.OPT and BG.FUT/BG.OPT return "Could not resolve smart symbols").
    However, the underlying data DOES exist and is queryable via raw_symbol
    (confirmed: a direct raw_symbol query for known Gasoil contracts
    succeeded). This means Gasoil simply isn't registered in Databento's
    parent-symbol index — a gap in their symbol lookup table, not in the data
    itself. The fix: enumerate real contract symbols by pulling ONE day's
    full instrument universe (via the special "ALL_SYMBOLS" raw_symbol query)
    and filtering locally for symbols starting with "F8", rather than relying
    on parent resolution at all for this instrument. See
    get_gasoil_symbols_via_all_symbols() below.

IMPORTANT — before running:
1. pip install databento pandas numpy scipy --break-system-packages
2. Get your API key from the Databento portal (API keys page).
3. This pulls the Statistics schema (official settlement price + open
   interest) and Definition schema (strike/expiry/underlying per contract) —
   NOT the raw tick/order-book schemas, which are unnecessarily heavy for
   this purpose.
4. Databento does not sell implied volatility as a field. This script
   computes it itself by inverting the Black-76 formula against the pulled
   settlement price — reusing the exact same pricing function you'll use
   for your benchmark model, just solved in reverse.
5. I have not been able to run or test this script myself (Databento's
   domain isn't reachable from my sandboxed environment) — treat this as a
   solid first draft to run and debug on your end, not verified-working code.
"""

import os
import databento as db
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------

# API key is read from an environment variable, NEVER hardcoded here — this
# file is meant to go into a public/supervisor-visible GitHub repo, and a
# hardcoded key would be exposed to anyone who sees it. Set it before running:
#   export DATABENTO_API_KEY="db-..."
# or place it in a local .env file (see .env.example) and load it with
# python-dotenv (pip install python-dotenv) by uncommenting the two lines below.
#
# from dotenv import load_dotenv
# load_dotenv()

API_KEY = os.environ.get("DATABENTO_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "DATABENTO_API_KEY environment variable not set. "
        "Run: export DATABENTO_API_KEY='db-your-key-here' before running this script."
    )

# NOTE: Both Brent and Gasoil are confirmed on the same dataset (GLBX.MDP3 —
# CME Globex), so the per-instrument "dataset" field below is redundant for
# now, but kept in case you need to add an instrument on a different dataset
# later.

INSTRUMENTS = {
    "brent": {
        "dataset": "GLBX.MDP3",  # CME Globex
        "option_root": "BE",     # confirmed: Brent Last Day Financial European Option
        "futures_root": "BZ",    # confirmed via screenshot (underlying e.g. BZH7, BZG7)
    },
    "gasoil": {
        "dataset": "GLBX.MDP3",  # CME Globex — confirmed directly on the F8 product page
        "option_root": "F8",     # confirmed: European-Style Low Sulphur Gasoil Option
        "futures_root": "BG",    # NOTE: confirmed WRONG as a parent-symbol root (BG.FUT
                                 # fails to resolve) — kept only as a label; not used as
                                 # a parent query for this instrument anymore.
        "needs_all_symbols_fallback": True,  # F8 isn't in Databento's parent-symbol
                                              # index — see get_gasoil_symbols_via_all_symbols
    },
}

WINDOWS = {
    "train_prior_shock": {"start": "2025-09-01", "end": "2025-12-31"},
    "train_calm":        {"start": "2026-01-01", "end": "2026-02-15"},
    "test_current_shock": {"start": "2026-05-01", "end": "2026-06-30"},
}

RATE_ASSUMPTION = 0.045  # placeholder flat short-term rate; replace with a
                          # real SOFR series if you can source one alongside

OUTPUT_DIR = "./dissertation_data"


# ----------------------------------------------------------------------
# 2. DATABENTO CLIENT
# ----------------------------------------------------------------------

client = db.Historical(API_KEY)


def _to_df_with_ts_column(data):
    """
    Databento's to_df() sometimes returns the timestamp as the DataFrame's
    index (a DatetimeIndex) rather than a plain "ts_event" column — this
    caused a KeyError('ts_event') on the futures pull specifically. Reset
    the index if it's a DatetimeIndex so downstream code can always rely on
    a real "ts_event" column existing, regardless of which schema returned it.
    """
    df = data.to_df()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "index" in df.columns and "ts_event" not in df.columns:
            df = df.rename(columns={"index": "ts_event"})
    return df


def get_gasoil_symbols_via_all_symbols(dataset: str, root_prefix: str, sample_date: str):
    """
    Workaround for products (like Gasoil "F8") that aren't registered in
    Databento's parent/smart-symbol index, even though their data exists.

    Pulls the FULL instrument universe's definitions for a single day using
    the special "ALL_SYMBOLS" raw_symbol query, then filters locally for
    symbols starting with root_prefix (e.g. "F8"). This is a much larger
    single pull than our normal chunked approach, but it's only done ONCE
    per instrument (not per date-chunk), and only for one representative day
    — option chains don't change drastically day-to-day within a window, so
    this gives us a usable (if not perfectly exact) contract list to then
    query normally via raw_symbol for the full date range.

    NOTE: retries a few subsequent days if the first comes back empty — the
    initial version of this used the exact first day of each window as the
    sample date, which happened to be Labor Day / New Year's Day (exchange
    holidays with no trading activity), producing a false "0 contracts
    found". Trying a few days forward avoids landing on another holiday.
    """
    for offset in range(0, 10):
        try_date = (pd.Timestamp(sample_date) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        try_end = (pd.Timestamp(try_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  Enumerating full instrument universe on {try_date} to find {root_prefix}* contracts...")
        data = client.timeseries.get_range(
            dataset=dataset,
            schema="definition",
            symbols=["ALL_SYMBOLS"],
            stype_in="raw_symbol",
            start=try_date,
            end=try_end,
        )
        df = _to_df_with_ts_column(data)
        if df.empty:
            print(f"  No data at all on {try_date} (likely a holiday/closure) — trying next day.")
            continue
        matches = df[df["raw_symbol"].astype(str).str.startswith(root_prefix)]
        symbols = matches["raw_symbol"].dropna().unique().tolist()
        if symbols:
            print(f"  Found {len(symbols)} contracts starting with '{root_prefix}' on {try_date}.")
            return symbols
        print(f"  0 contracts starting with '{root_prefix}' on {try_date} — trying next day.")
    print(f"  WARNING: found no {root_prefix}* contracts across 10 days tried starting {sample_date}.")
    return []


def get_option_definitions(dataset: str, option_root: str, start: str, end: str, explicit_symbols: list = None):
    """
    Pull contract definitions (strike, expiry, underlying) for all
    instruments under a given option root over a date range. Chunked by
    date for the same timeout reason as get_settlement_and_oi.

    If explicit_symbols is provided (a list of raw contract symbols), this
    queries those directly via stype_in="raw_symbol" instead of using parent
    symbology — needed for products like Gasoil "F8" that aren't registered
    in Databento's parent-symbol index (see get_gasoil_symbols_via_all_symbols).
    """
    frames = []
    for chunk_start, chunk_end in date_chunks(start, end):
        try:
            if explicit_symbols:
                data = client.timeseries.get_range(
                    dataset=dataset,
                    schema="definition",
                    symbols=explicit_symbols,
                    stype_in="raw_symbol",
                    start=chunk_start,
                    end=chunk_end,
                )
            else:
                data = client.timeseries.get_range(
                    dataset=dataset,
                    schema="definition",
                    symbols=[f"{option_root}.OPT"],
                    stype_in="parent",
                    start=chunk_start,
                    end=chunk_end,
                )
            df_chunk = _to_df_with_ts_column(data)
            if not df_chunk.empty:
                frames.append(df_chunk)
        except Exception as e:
            print(f"  WARNING: definitions chunk {chunk_start}-{chunk_end} failed: {e}")
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["raw_symbol"])


def date_chunks(start: str, end: str, freq_days: int = 14):
    """
    Split a date range into smaller chunks (default ~2 weeks) to avoid
    server-side read timeouts on large option-chain requests. Databento's
    connection appears to time out on requests spanning several months for
    a large chain in one call — pulling in smaller pieces and concatenating
    is more reliable than one big request, even though it's more calls.
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    chunks = []
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + pd.Timedelta(days=freq_days), end_dt)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end
    return chunks


def get_settlement_and_oi(dataset: str, option_root: str, start: str, end: str, explicit_symbols: list = None):
    """
    Pull official daily settlement price and open interest for an entire
    option chain, using parent symbology (e.g. "BE.OPT") by default.

    If explicit_symbols is provided, queries those directly via
    stype_in="raw_symbol" instead — needed for products like Gasoil "F8"
    that aren't registered in Databento's parent-symbol index. Symbol lists
    are also chunked under the 2,000-symbols-per-request cap in this mode.

    NOTE: pulled in date chunks (see date_chunks) rather than one request for
    the whole window, since a single large request timed out in practice
    (BentoError: Read timed out) — the full Brent chain over several months
    is too much data for one streaming call to complete reliably.
    """
    frames = []

    def _symbol_batches(symbols, batch_size=1900):
        for i in range(0, len(symbols), batch_size):
            yield symbols[i:i + batch_size]

    for chunk_start, chunk_end in date_chunks(start, end):
        print(f"  ...pulling statistics {chunk_start} to {chunk_end}")
        try:
            if explicit_symbols:
                for batch in _symbol_batches(explicit_symbols):
                    data = client.timeseries.get_range(
                        dataset=dataset,
                        schema="statistics",
                        symbols=batch,
                        stype_in="raw_symbol",
                        start=chunk_start,
                        end=chunk_end,
                    )
                    df_chunk = _to_df_with_ts_column(data)
                    if not df_chunk.empty:
                        frames.append(df_chunk)
            else:
                data = client.timeseries.get_range(
                    dataset=dataset,
                    schema="statistics",
                    symbols=[f"{option_root}.OPT"],
                    stype_in="parent",
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
    return pd.concat(frames, ignore_index=True)


def get_futures_settlement(dataset: str, futures_root: str, start: str, end: str, explicit_symbols: list = None):
    """
    Pull daily settlement prices for the futures strip under a given root,
    needed to match each option to its correct underlying expiry.

    If explicit_symbols is provided (real underlying futures symbols, e.g.
    "BGX2", pulled from the option definitions themselves), queries those
    directly via stype_in="raw_symbol" instead of parent symbology — needed
    because Gasoil's futures root ("BG") also fails to resolve as a parent
    symbol, same issue as the option root "F8".
    """
    frames = []
    for chunk_start, chunk_end in date_chunks(start, end):
        try:
            if explicit_symbols:
                data = client.timeseries.get_range(
                    dataset=dataset,
                    schema="ohlcv-1d",
                    symbols=explicit_symbols,
                    stype_in="raw_symbol",
                    start=chunk_start,
                    end=chunk_end,
                )
            else:
                data = client.timeseries.get_range(
                    dataset=dataset,
                    schema="ohlcv-1d",
                    symbols=[f"{futures_root}.FUT"],
                    stype_in="parent",
                    start=chunk_start,
                    end=chunk_end,
                )
            df_chunk = _to_df_with_ts_column(data)
            if not df_chunk.empty:
                frames.append(df_chunk)
        except Exception as e:
            print(f"  WARNING: futures chunk {chunk_start}-{chunk_end} failed: {e}")
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# 3. BLACK-76 PRICING AND IMPLIED VOL INVERSION
# ----------------------------------------------------------------------

def black76_price(F, K, T, r, sigma, option_type="C"):
    """Black-76 price for a European option on a futures contract."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (F - K) if option_type == "C" else (K - F))
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if option_type == "C":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, r, option_type="C"):
    """
    Back out implied volatility by inverting black76_price via root-finding.
    Reuses the same pricing function as the Black-76 benchmark model.
    """
    if price <= 0 or T <= 0:
        return np.nan

    def objective(sigma):
        return black76_price(F, K, T, r, sigma, option_type) - price

    try:
        return brentq(objective, 1e-6, 5.0, xtol=1e-6)
    except ValueError:
        return np.nan  # no sign change in bracket — price outside no-arbitrage bounds


# ----------------------------------------------------------------------
# 4. MERGE + FEATURE CONSTRUCTION
# ----------------------------------------------------------------------

def build_dataset(instrument_name: str, config: dict, window_name: str, dates: dict):
    print(f"\n--- {instrument_name} | {window_name} ---")

    dataset = config["dataset"]

    # For products not registered in Databento's parent-symbol index (see
    # Gasoil "F8" note at top of file), enumerate real contract symbols via
    # a one-day ALL_SYMBOLS query first, then query defs/stats with those
    # explicit symbols instead of relying on parent resolution.
    explicit_symbols = None
    if config.get("needs_all_symbols_fallback"):
        explicit_symbols = get_gasoil_symbols_via_all_symbols(
            dataset, config["option_root"], dates["start"],
        )
        if not explicit_symbols:
            print(f"No {config['option_root']}* contracts found via ALL_SYMBOLS fallback.")
            return pd.DataFrame()

    defs = get_option_definitions(
        dataset, config["option_root"], dates["start"], dates["end"],
        explicit_symbols=explicit_symbols,
    )
    if defs.empty:
        print("No definitions returned — check option_root / date range.")
        return pd.DataFrame()

    stats = get_settlement_and_oi(
        dataset, config["option_root"], dates["start"], dates["end"],
        explicit_symbols=explicit_symbols,
    )

    # For the fallback-mode instruments (Gasoil), the futures root also
    # fails to resolve as a parent symbol — so instead pull the real
    # underlying futures symbols directly out of the option definitions
    # (e.g. "BGX2") and query those explicitly, rather than using
    # config["futures_root"] as a parent query.
    if config.get("needs_all_symbols_fallback"):
        underlying_symbols = defs["underlying"].dropna().unique().tolist()
        futures = get_futures_settlement(
            dataset, config["futures_root"], dates["start"], dates["end"],
            explicit_symbols=underlying_symbols,
        )
    else:
        futures = get_futures_settlement(dataset, config["futures_root"], dates["start"], dates["end"])

    if stats.empty:
        print("No statistics returned — check option_root / date range.")
        return pd.DataFrame()

    # --- Keep only option instruments (not the parent futures also present
    # under the same root), and the definition fields we actually need.
    # Following Databento's own documented pattern (see their "Volume, open
    # interest, and settlement prices" example): keep one definition row per
    # instrument_id (defs can otherwise contain multiple historical
    # revisions per contract).
    option_classes = {"C", "P"}  # Call / Put — adjust if defs show different codes
    defs_opt = defs[defs["instrument_class"].isin(option_classes)].copy()
    defs_opt = defs_opt[[
        "instrument_id", "raw_symbol", "strike_price", "expiration",
        "instrument_class", "underlying",
    ]].drop_duplicates(subset=["instrument_id"], keep="last")

    # --- Merge statistics with definitions on instrument_id (NOT raw_symbol
    # — statistics rows don't carry raw_symbol, only instrument_id/symbol).
    merged = stats.merge(defs_opt, on="instrument_id", how="left")

    # --- The statistics schema is long-format: each row is ONE stat_type
    # (e.g. settlement price OR open interest OR cleared volume), not a wide
    # row with all three as separate columns. Pull each stat type out into
    # its own column, then collapse to one row per (date, instrument).
    # NOTE: use ts_event (always populated), not ts_ref — an earlier version
    # preferred ts_ref when present, but it was mostly null, which silently
    # dropped every row in the groupby below (pandas drops null-keyed groups
    # by default). This was the actual cause of the "0 rows saved" bug.
    merged["trade_date"] = pd.to_datetime(merged["ts_event"]).dt.date

    merged["settlement_price"] = merged.loc[merged["stat_type"] == db.StatType.SETTLEMENT_PRICE, "price"]
    merged["open_interest"] = merged.loc[merged["stat_type"] == db.StatType.OPEN_INTEREST, "quantity"]
    merged["cleared_volume"] = merged.loc[merged["stat_type"] == db.StatType.CLEARED_VOLUME, "quantity"]

    print(f"  DEBUG rows going into groupby: {len(merged)}, "
          f"non-null trade_date: {merged['trade_date'].notna().sum()}, "
          f"non-null settlement_price: {merged['settlement_price'].notna().sum()}")

    merged = (
        merged.groupby(["trade_date", "raw_symbol"], as_index=False)
        .agg({
            "settlement_price": "last",
            "open_interest": "last",
            "cleared_volume": "last",
            "strike_price": "last",
            "expiration": "last",
            "instrument_class": "last",
            "underlying": "last",
        })
    )

    # A (date, contract) row is only meaningful if it has a real settlement
    # price — this is the actual "did this contract have a value today"
    # signal in this feed. Open interest is reported far more sparsely
    # (confirmed: only a few hundred OI records across thousands of
    # contracts and months of data) and should NOT be used as a hard
    # liquidity filter — treating a missing OI as "illiquid, drop" was
    # discarding almost all genuinely good rows. Drop only on missing price.
    before_price_filter = len(merged)
    merged = merged[merged["settlement_price"].notna()].copy()
    print(f"  DEBUG dropped {before_price_filter - len(merged)} rows with no settlement_price "
          f"({len(merged)} remain)")

    # Merge in the matching futures settlement price by expiry month.
    # NOTE: exact column names from the futures ohlcv-1d pull (e.g. "symbol",
    # "close") are assumed from Databento's standard OHLCV schema — verify
    # against real output if this merge silently produces all-NaN futures_settle.
    if not futures.empty:
        futures = futures.copy()
        futures["trade_date"] = pd.to_datetime(futures["ts_event"]).dt.date
        merged = merged.merge(
            futures[["trade_date", "symbol", "close"]].rename(columns={"close": "futures_settle", "symbol": "underlying"}),
            on=["trade_date", "underlying"],
            how="left",
        )
    else:
        merged["futures_settle"] = np.nan

    merged["ttm_years"] = (
        pd.to_datetime(merged["expiration"]).dt.tz_localize(None)
        - pd.to_datetime(merged["trade_date"])
    ).dt.days / 365.25

    merged["option_type"] = merged["instrument_class"].map(
        lambda x: "C" if str(x).upper().startswith("C") else "P"
    )

    merged["implied_vol"] = merged.apply(
        lambda row: implied_vol(
            price=row.get("settlement_price", np.nan),
            F=row.get("futures_settle", np.nan),
            K=row.get("strike_price", np.nan),
            T=row.get("ttm_years", np.nan),
            r=RATE_ASSUMPTION,
            option_type=row.get("option_type", "C"),
        ),
        axis=1,
    )

    merged["instrument"] = instrument_name
    merged["window"] = window_name
    return merged


def filter_liquid(df: pd.DataFrame, min_oi: int = 10):
    """
    Optional secondary liquidity filter — drops a row ONLY if open_interest
    is present AND below the threshold. A missing/NaN open_interest is
    treated as "unknown", not "zero", and is kept rather than dropped.

    NOTE: given how sparsely open_interest is actually reported in this
    feed (confirmed: a few hundred records across thousands of contracts
    over months of data), this filter will affect very few rows either way
    — the real "does this row matter" filtering already happened in
    build_dataset by requiring a non-null settlement_price. This function is
    kept as a light secondary pass, not the primary liquidity gate.
    """
    if "open_interest" not in df.columns:
        return df
    has_oi = df["open_interest"].notna()
    return df[~has_oi | (df["open_interest"] >= min_oi)]


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for instrument_name, config in INSTRUMENTS.items():
        if "CONFIRM_ME" in (config["option_root"], config["futures_root"]):
            print(f"Skipping {instrument_name} — ticker not yet confirmed.")
            continue

        for window_name, dates in WINDOWS.items():
            df = build_dataset(instrument_name, config, window_name, dates)
            if df.empty:
                continue
            df = filter_liquid(df)
            out_path = f"{OUTPUT_DIR}/{instrument_name}_{window_name}.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved {out_path} ({len(df)} rows)")

    print("\nDone. Next steps:")
    print("1. Sanity-check the futures-matching merge — 'symbol'/'close' column")
    print("   names for the futures ohlcv-1d pull are assumed from Databento's")
    print("   standard schema; verify against real output if futures_settle is")
    print("   all-NaN.")
    print("2. Replace RATE_ASSUMPTION with a real short-term rate series once sourced.")
    print("3. Validate implied_vol output against a few known option prices by hand.")
