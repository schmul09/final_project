import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# STANDALONE VERSION — loads and combines the three labelled raw files.
# ----------------------------------------------------------------------

prior_shock = pd.read_csv("data/raw/brent_train_prior_shock_labelled.csv")
prior_shock["window"] = "train_prior_shock"

calm = pd.read_csv("data/raw/brent_train_calm_labelled.csv")
calm["window"] = "train_calm"

current_shock = pd.read_csv("data/raw/brent_test_current_shock_labelled.csv")
current_shock["window"] = "test_current_shock"

all_data = pd.concat([prior_shock, calm, current_shock], ignore_index=True)
print(f"Loaded and combined {len(all_data)} total rows across all three windows")
print(all_data["window"].value_counts())

# Sanity check: confirm the columns needed below actually exist before continuing
required_cols = ["settlement_price", "strike_price", "futures_settle",
                  "option_type", "moneyness"]
missing_cols = [c for c in required_cols if c not in all_data.columns]
if missing_cols:
    print(f"\nWARNING: expected columns not found: {missing_cols}")
    print(f"Actual columns in the file: {list(all_data.columns)}")
    raise SystemExit("Fix column names above before continuing.")

# Recompute intrinsic value and the violation mask (same logic as the main script)
intrinsic_call = (all_data["futures_settle"] - all_data["strike_price"]).clip(lower=0)
intrinsic_put = (all_data["strike_price"] - all_data["futures_settle"]).clip(lower=0)
intrinsic = np.where(all_data["option_type"] == "C", intrinsic_call, intrinsic_put)

TOLERANCE = 0.5
violation = all_data["settlement_price"] < (intrinsic - TOLERANCE)

# ----------------------------------------------------------------------
# MANUAL INSPECTION: no-arbitrage bound violations
# ----------------------------------------------------------------------

violated = all_data[violation].copy()
violated["shortfall"] = intrinsic[violation] - violated["settlement_price"]

print(f"\n=== Inspecting {len(violated)} no-arbitrage violations ===\n")

print("Shortfall distribution (amount below the theoretical floor):")
print(violated["shortfall"].describe())
print(f"\nRows with shortfall > $5 (likely genuine errors): "
      f"{(violated['shortfall'] > 5).sum()}")
print(f"Rows with shortfall between tolerance and $2 (likely stale/illiquid quotes): "
      f"{((violated['shortfall'] > 0.5) & (violated['shortfall'] <= 2)).sum()}")

print("\nViolations by window:")
print(violated["window"].value_counts())
print("\nShare of each window's rows that are violations:")
print((violated["window"].value_counts() / all_data["window"].value_counts()).round(3))

instrument_col = "instrument" if "instrument" in all_data.columns else None
if instrument_col:
    print("\nTop 10 instruments by violation count:")
    print(violated[instrument_col].value_counts().head(10))
else:
    print("\n(No 'instrument' column found — skipping per-contract breakdown. "
          "Check actual column name for contract identifier.)")

print("\nMoneyness distribution of violating rows vs. full dataset:")
print("Violations:", violated["moneyness"].describe()[["mean", "50%", "min", "max"]])
print("Full data: ", all_data["moneyness"].describe()[["mean", "50%", "min", "max"]])

large_shortfall = violated["shortfall"] > 5
print(f"\nSuggested to DROP as likely errors: {large_shortfall.sum()} rows")
print(f"Suggested to KEEP (flagged) as likely stale quotes: "
      f"{(~large_shortfall).sum()} rows")