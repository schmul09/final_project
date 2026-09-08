"""
Produces final, model-ready datasets from the labelled CSVs:
  - Adds moneyness and realized_vol as permanent columns (previously only
    computed transiently inside other scripts, never saved).
  - Filters to rows that have BOTH a valid hedge_ratio_label (from
    construct_hedge_label.py) AND complete values for all four model
    inputs (moneyness, ttm_years, realized_vol, rate).
  - Splits calls and puts into separate files, since separate networks
    are being trained for each (see the output-activation decision in
    Methodology 3.1.5).

Run from the project root: python src/finalize_model_data.py
"""

import pandas as pd
import numpy as np

FILES = {
    "train_prior_shock": "data/raw/brent_train_prior_shock_clean_labelled.csv",
    "train_calm": "data/raw/brent_train_calm_clean_labelled.csv",
    "test_current_shock": "data/raw/brent_test_current_shock_clean_labelled.csv",
}

OUTPUT_DIR = "data/model_ready"

FEATURES = ["moneyness", "ttm_years", "realized_vol", "rate"]


def add_moneyness_and_realized_vol(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["underlying", "trade_date"]).copy()

    df["moneyness"] = df["strike_price"] / df["futures_settle"]

    # 20-day rolling realised volatility of the underlying futures price,
    # computed per underlying contract month (same construction used in
    # the multicollinearity check, now made permanent).
    df["log_return"] = (
        df.groupby("underlying")["futures_settle"]
        .apply(lambda s: np.log(s / s.shift(1)))
        .reset_index(level=0, drop=True)
    )
    df["realized_vol"] = (
        df.groupby("underlying")["log_return"]
        .transform(lambda s: s.rolling(20).std() * np.sqrt(252))
    )
    return df


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for window_name, path in FILES.items():
        print(f"\n--- {window_name} ---")
        df = pd.read_csv(path, parse_dates=["trade_date"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        df = add_moneyness_and_realized_vol(df)

        # Keep only rows with a valid label AND complete features.
        # realized_vol will be NaN for the first 20 observations of each
        # underlying contract month (nothing to compute a rolling window
        # from yet) — a real, expected source of missingness distinct from
        # the ones already documented, worth noting in the write-up.
        before = len(df)
        has_label = df["hedge_ratio_label"].notna()
        has_features = df[FEATURES].notna().all(axis=1)
        has_reasonable_label = df["hedge_ratio_label"].abs() <= 2.0
        df_final = df[has_label & has_features & has_reasonable_label].copy()

        print(f"  Total rows: {before}")
        print(f"  Missing label: {(~has_label).sum()}")
        print(f"  Missing one or more features (mostly realized_vol warm-up): "
              f"{(has_label & ~has_features).sum()}")
        print(f"  Final usable rows: {len(df_final)} ({len(df_final)/before*100:.1f}%)")

        for opt_type, label_name in [("C", "calls"), ("P", "puts")]:
            subset = df_final[df_final["option_type"] == opt_type]
            out_path = f"{OUTPUT_DIR}/{window_name}_{label_name}.csv"
            subset.to_csv(out_path, index=False)
            print(f"  Saved {out_path} ({len(subset)} rows)")

    print("\nDone. Model-ready files are in data/model_ready/, split by "
          "window and option type.")