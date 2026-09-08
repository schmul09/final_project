"""
Data preparation for the Brent options dataset:
loading, missing-value handling, outlier handling, and the
train / validation / test split.

IMPORTANT — this is NOT a standard IID dataset, and standard exam-style
cleaning rules need adapting:
  1. Missing values here are often NOT random — e.g. open_interest is
     genuinely sparsely reported in this feed (not "dirty data" to impute).
  2. Extreme values during the shock windows are not necessarily noise —
     they may be the actual phenomenon the dissertation studies. Aggressive
     outlier removal could delete the most important observations.
  3. The train/val/test split MUST be chronological, not random — see the
     rationale in Section 4.
"""

import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------

DATA_DIR = "data/raw"

files = {
    "train_prior_shock": f"{DATA_DIR}/brent_train_prior_shock.csv",
    "train_calm": f"{DATA_DIR}/brent_train_calm.csv",
    "test_current_shock": f"{DATA_DIR}/brent_test_current_shock.csv",
}

dfs = {}
for window_name, path in files.items():
    df = pd.read_csv(path, parse_dates=["trade_date", "expiration"])
    df["window"] = window_name
    dfs[window_name] = df
    print(f"{window_name}: {len(df)} rows loaded")

# Combined view for EDA (NOT for training directly — see Section 4 for the
# actual train/val/test assembly, which keeps windows separate)
all_data = pd.concat(dfs.values(), ignore_index=True)
print(f"\nTotal rows across all windows: {len(all_data)}")


# ----------------------------------------------------------------------
# 2. INITIAL INSPECTION
# ----------------------------------------------------------------------

print("\n--- Column dtypes ---")
print(all_data.dtypes)

print("\n--- Missing value counts (and %) ---")
missing = all_data.isna().sum()
missing_pct = (missing / len(all_data) * 100).round(1)
print(pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}))

print("\n--- Summary statistics (numeric columns) ---")
print(all_data.describe())


# ----------------------------------------------------------------------
# 3. MISSING VALUES
# ----------------------------------------------------------------------

# settlement_price: should already be non-null (filtered during extraction),
# but double-check — a missing price makes a row unusable regardless.
before = len(all_data)
all_data = all_data[all_data["settlement_price"].notna()]
print(f"\nDropped {before - len(all_data)} rows with missing settlement_price")

# futures_settle: CRITICAL — needed for moneyness, Black-76 pricing, and the
# NN's core input feature. A row without it cannot be used at all.
before = len(all_data)
all_data = all_data[all_data["futures_settle"].notna()]
print(f"Dropped {before - len(all_data)} rows with missing futures_settle "
      f"(this is the merge-quality check — a large number here would "
      f"indicate the futures-matching step needs revisiting)")

# implied_vol: can be NaN if the Black-76 inversion failed (e.g. price
# outside no-arbitrage bounds, or extreme moneyness/TTM combinations).
# These rows are excluded from vol-dependent modelling but NOT necessarily
# dropped from the whole dataset — flag rather than delete outright, since
# you may still want the raw price data for some analyses.
all_data["has_valid_iv"] = all_data["implied_vol"].notna()
print(f"\n{all_data['has_valid_iv'].sum()} of {len(all_data)} rows have a "
      f"valid implied_vol; {(~all_data['has_valid_iv']).sum()} do not "
      f"(flagged via has_valid_iv, not dropped)")

# open_interest / cleared_volume: genuinely sparse in this feed (confirmed
# during extraction — only a few hundred records across thousands of
# contracts). Do NOT impute with 0 (that would misrepresent "unknown" as
# "no interest") and do NOT use as a hard filter. Leave as NaN; downstream
# code should treat missing OI as "unknown", not "illiquid".
print(f"\nopen_interest populated for {all_data['open_interest'].notna().sum()} "
      f"of {len(all_data)} rows ({all_data['open_interest'].notna().mean()*100:.1f}%) "
      f"— left as NaN elsewhere, not imputed, given how sparse this field is")


# ----------------------------------------------------------------------
# 4. OUTLIER HANDLING
# ----------------------------------------------------------------------

# IMPORTANT: standard IQR/z-score outlier removal is risky here. The shock
# windows are DEFINED by extreme price/vol movements — aggressively removing
# "outliers" could delete exactly the observations the dissertation is about.
# Distinguish between two different things:
#   (a) genuinely impossible/corrupted values (data errors) — remove these
#   (b) genuinely extreme but real market values during a shock — keep these,
#       but consider flagging them for sensitivity analysis in the discussion

# (a) Hard, non-negotiable sanity filters — these indicate data errors, not
# real extreme markets, and should always be removed:
before = len(all_data)
all_data = all_data[all_data["settlement_price"] > 0]  # price cannot be zero/negative
all_data = all_data[all_data["strike_price"] > 0]
all_data = all_data[all_data["ttm_years"] > 0]           # already-expired contracts
print(f"\nDropped {before - len(all_data)} rows failing basic sanity checks "
      f"(non-positive price/strike/TTM)")

# No-arbitrage bound check: a call price cannot exceed the futures price
# (deep discount aside), and cannot be negative relative to intrinsic value
# by more than a small tolerance. Large violations suggest a data/merge
# error rather than a real market price.
all_data["moneyness"] = all_data["strike_price"] / all_data["futures_settle"]

intrinsic_call = (all_data["futures_settle"] - all_data["strike_price"]).clip(lower=0)
intrinsic_put = (all_data["strike_price"] - all_data["futures_settle"]).clip(lower=0)
intrinsic = np.where(all_data["option_type"] == "C", intrinsic_call, intrinsic_put)

TOLERANCE = 0.5  # allow some slack for discounting/rounding — not zero
violation = all_data["settlement_price"] < (intrinsic - TOLERANCE)
print(f"\n{violation.sum()} rows violate the no-arbitrage lower bound by more "
      f"than the tolerance — inspect these manually before deciding whether "
      f"to drop them (could be data errors, or genuinely stale/illiquid quotes)")
before = len(all_data)
all_data = all_data[~violation]
print(f"Dropped {before - len(all_data)} rows violating the no-arbitrage bound")


# (b) Extreme-but-real values — FLAG, don't drop, especially in shock windows:
# moneyness far from 1.0 (deep ITM/OTM) is normal for an option chain, not
# an error — only worth flagging for awareness, not removing.
extreme_moneyness = (all_data["moneyness"] < 0.5) | (all_data["moneyness"] > 2.0)
print(f"\n{extreme_moneyness.sum()} rows have moneyness outside [0.5, 2.0] "
      f"(deep ITM/OTM) — kept in the dataset; these are a normal, expected "
      f"part of an option chain, not outliers to remove")


# ----------------------------------------------------------------------
# 5. TRAIN / VALIDATION / TEST SPLIT
# ----------------------------------------------------------------------

# NOT a random split. Two reasons:
#   1. Same-day observations aren't independent (shared underlying price,
#      shared market conditions) — random splitting would leak information
#      between train and validation.
#   2. The research question is specifically about generalising to a NEW
#      shock episode, so the test set must be a held-out TIME PERIOD, not a
#      random sample of rows.

# Test set: the entire current-shock window, untouched.
test_df = all_data[all_data["window"] == "test_current_shock"].copy()

# Train/validation pool: prior-shock + calm windows combined.
train_pool = all_data[all_data["window"].isin(["train_prior_shock", "train_calm"])].copy()

# Within the pool, take the most recent ~15-20% of DATES (not rows) from
# EACH regime block separately as validation — this keeps validation
# representative of both regimes, rather than accidentally all-calm or
# all-shock if one block happens to sit at the very end of the timeline.
VALIDATION_FRACTION = 0.15

val_frames = []
train_frames = []
for window_name in ["train_prior_shock", "train_calm"]:
    block = train_pool[train_pool["window"] == window_name]
    unique_dates = sorted(block["trade_date"].unique())
    n_val_dates = max(1, int(len(unique_dates) * VALIDATION_FRACTION))
    val_dates = set(unique_dates[-n_val_dates:])  # most recent dates in this block

    val_frames.append(block[block["trade_date"].isin(val_dates)])
    train_frames.append(block[~block["trade_date"].isin(val_dates)])

val_df = pd.concat(val_frames, ignore_index=True)
train_df = pd.concat(train_frames, ignore_index=True)

print(f"\n--- Final split ---")
print(f"Train: {len(train_df)} rows, {train_df['trade_date'].nunique()} unique dates")
print(f"Validation: {len(val_df)} rows, {val_df['trade_date'].nunique()} unique dates")
print(f"Test (held-out shock): {len(test_df)} rows, {test_df['trade_date'].nunique()} unique dates")

# Sanity check: confirm no date overlap between train and validation within
# the same window (would indicate a bug in the split logic above).
train_dates_by_window = train_df.groupby("window")["trade_date"].apply(set)
val_dates_by_window = val_df.groupby("window")["trade_date"].apply(set)
for w in train_dates_by_window.index:
    overlap = train_dates_by_window[w] & val_dates_by_window.get(w, set())
    if overlap:
        print(f"WARNING: {len(overlap)} overlapping dates between train/val in {w}")
    else:
        print(f"OK: no train/val date overlap in {w}")


# ----------------------------------------------------------------------
# 6. SAVE CLEANED DATA (per window) so downstream scripts — construct_label.py
#    in particular — read the cleaned/filtered data rather than the original
#    raw files. This is the step that was previously missing: everything
#    above this point was computed in memory and then discarded.
# ----------------------------------------------------------------------

for window_name in ["train_prior_shock", "train_calm", "test_current_shock"]:
    window_df = all_data[all_data["window"] == window_name]
    out_path = f"{DATA_DIR}/brent_{window_name}_clean.csv"
    window_df.to_csv(out_path, index=False)
    print(f"Saved {len(window_df)} cleaned rows to {out_path}")

print("\nDone. Update construct_label.py's FILES dictionary to point at "
      "these *_clean.csv files instead of the original raw files, then "
      "re-run the pipeline from construct_label.py onward.")