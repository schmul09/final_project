"""
Construct the empirical local hedge-ratio label described in Data section
2.3.5: for each contract (same raw_symbol), the label at observation t is
the realised change in option price divided by the realised change in the
underlying futures price, measured against the NEXT observation of that
same contract (t -> t+1), then assigned back to the observation at t.

This gives the network a data-driven analogue of delta to learn from,
rather than one derived from Black-76's formula.

Run from the project root: python src/construct_hedge_label.py
"""

import pandas as pd
import numpy as np

FILES = {
    "train_prior_shock": "data/raw/brent_train_prior_shock_clean.csv",
    "train_calm": "data/raw/brent_train_calm_clean.csv",
    "test_current_shock": "data/raw/brent_test_current_shock_clean.csv",
}

# Minimum futures price move required to compute a label. Below this,
# dividing by a near-zero futures move can produce an exploding,
# economically meaningless ratio (e.g. a tiny ΔF with a modest ΔC gives a
# huge "hedge ratio" that doesn't reflect real hedging behaviour). This is
# an outlier-prevention step, following the same principle as the
# no-arbitrage sanity checks in data_preparation.py: a hard rule for
# genuine data-artefact risk, not a judgement call on real market moves.
MIN_ABS_DELTA_F = 0.5  # USD — small relative to typical daily Brent moves


def construct_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["raw_symbol", "trade_date"]).copy()

    # Next observation of the SAME contract (raw_symbol uniquely identifies
    # one specific strike/expiry/type, so grouping by it is what makes this
    # a same-contract comparison rather than a same-day, different-contract
    # comparison).
    df["next_settlement_price"] = df.groupby("raw_symbol")["settlement_price"].shift(-1)
    df["next_futures_settle"] = df.groupby("raw_symbol")["futures_settle"].shift(-1)
    df["next_trade_date"] = df.groupby("raw_symbol")["trade_date"].shift(-1)

    df["delta_C"] = df["next_settlement_price"] - df["settlement_price"]
    df["delta_F"] = df["next_futures_settle"] - df["futures_settle"]
    df["gap_days"] = (df["next_trade_date"] - df["trade_date"]).dt.days

    # Rows with no next observation (e.g. the contract's last recorded day,
    # or it stopped trading) cannot get a label — this is a new missing-
    # data cause distinct from the ones already documented in 2.3.3.
    has_next_obs = df["next_settlement_price"].notna()

    # Guard against division by a near-zero futures move.
    valid_delta_f = df["delta_F"].abs() >= MIN_ABS_DELTA_F

    df["hedge_ratio_label"] = np.where(
        has_next_obs & valid_delta_f,
        df["delta_C"] / df["delta_F"],
        np.nan,
    )

    n_total = len(df)
    n_no_next = (~has_next_obs).sum()
    n_tiny_delta_f = (has_next_obs & ~valid_delta_f).sum()
    n_labelled = df["hedge_ratio_label"].notna().sum()

    print(f"  Total rows: {n_total}")
    print(f"  No next observation for this contract: {n_no_next}")
    print(f"  Next observation exists but |ΔF| < {MIN_ABS_DELTA_F}: {n_tiny_delta_f}")
    print(f"  Successfully labelled: {n_labelled} ({n_labelled/n_total*100:.1f}%)")

    return df


if __name__ == "__main__":
    for window_name, path in FILES.items():
        print(f"\n--- {window_name} ---")
        df = pd.read_csv(path, parse_dates=["trade_date"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        df = construct_labels(df)

        # Quick sanity check on the labelled values themselves — a call's
        # hedge ratio should sit roughly in [0, 1], a put's in [-1, 0].
        # Real-world noise means some will fall outside this exactly, but
        # a systematic pattern of wildly-out-of-range values would signal
        # a construction bug worth investigating before training on this.
        labelled = df[df["hedge_ratio_label"].notna()]
        print(f"  Gap between matched observations (calendar days): "
              f"min={labelled['gap_days'].min():.0f}, "
              f"median={labelled['gap_days'].median():.0f}, "
              f"max={labelled['gap_days'].max():.0f}")

        for opt_type, expected_range in [("C", (0, 1)), ("P", (-1, 0))]:
            subset = labelled[labelled["option_type"] == opt_type]["hedge_ratio_label"]
            if len(subset) > 0:
                pct_in_range = ((subset >= expected_range[0] - 0.1) &
                                 (subset <= expected_range[1] + 0.1)).mean() * 100
                print(f"  {opt_type}: {len(subset)} labelled, "
                      f"{pct_in_range:.1f}% within expected range {expected_range} (±0.1 tolerance)")

                in_range_mask = ((subset >= expected_range[0] - 0.1) &
                                  (subset <= expected_range[1] + 0.1))
                gaps_in_range = labelled.loc[subset[in_range_mask].index, "gap_days"]
                gaps_out_of_range = labelled.loc[subset[~in_range_mask].index, "gap_days"]
                if len(gaps_out_of_range) > 0:
                    print(f"      median gap, in-range: {gaps_in_range.median():.0f} days | "
                          f"out-of-range: {gaps_out_of_range.median():.0f} days")

        out_path = path.replace(".csv", "_labelled.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")

    print("\nDone. Review the *_labelled.csv files — check the range-check "
          "percentages above before treating these labels as reliable.")