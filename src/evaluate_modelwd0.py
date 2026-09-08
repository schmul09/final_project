"""
Headline evaluation (Section 3.3 / Results): compares the trained neural
network's hedge-ratio predictions against Black-76's analytical delta,
both evaluated against what ACTUALLY happened in the market on the
held-out test_current_shock window.

The comparison metric is squared hedging error: for each observation,
(realised ΔC - predicted_hedge_ratio × realised ΔF)^2 — i.e. how much
P&L variance would have remained if you'd hedged using that model's
ratio, given what the option and futures price actually did next. The
mean of this across all test rows is the Mean Squared Hedging Error
(MSHE). Lower is better for both models; the comparison between them is
the dissertation's central empirical result.

Run from the project root: python src/evaluate_model.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

FEATURES = ["moneyness", "ttm_years", "realized_vol", "rate"]
CHECKPOINT_DIR = "checkpoints_4features_wd0"
TEST_WINDOW = "test_current_shock"
MODEL_READY_DIR = "data/model_ready"


# Same architecture as train_model.py — duplicated here rather than
# imported, to keep this script runnable standalone regardless of working
# directory (imports between sibling scripts have been a repeated source
# of path issues in this project).
class HedgeRatioNet(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: list, option_type: str):
        super().__init__()
        self.option_type = option_type
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
            in_size = h
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_size, 1)

    def forward(self, x):
        h = self.hidden(x)
        raw_out = self.output_layer(h)
        bounded = torch.sigmoid(raw_out)
        if self.option_type == "P":
            bounded = -bounded
        return bounded.squeeze(-1)


def black76_delta(F, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    disc = np.exp(-r * T)
    if option_type == "C":
        return disc * norm.cdf(d1)
    return disc * (norm.cdf(d1) - 1)  # equivalent to -disc * norm.cdf(-d1)


def evaluate_one_type(option_type: str, label_name: str):
    print(f"\n{'='*60}\nEvaluating {label_name} (option_type={option_type})\n{'='*60}")

    checkpoint = torch.load(f"{CHECKPOINT_DIR}/{label_name}_model.pt", weights_only=False)
    model = HedgeRatioNet(
        n_features=len(FEATURES),
        hidden_sizes=checkpoint["hidden_sizes"],
        option_type=checkpoint["option_type"],
    )
    print("hidden_sizes from checkpoint:", checkpoint["hidden_sizes"])
    print("Freshly-built model's state_dict keys:", list(model.state_dict().keys()))
    print("Checkpoint's state_dict keys:", list(checkpoint["model_state_dict"].keys()))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_mean = pd.Series(checkpoint["feature_mean"])
    feature_std = pd.Series(checkpoint["feature_std"])

    df = pd.read_csv(f"{MODEL_READY_DIR}/{TEST_WINDOW}_{label_name}.csv", parse_dates=["trade_date"])
    before = len(df)

    # Black-76 requires implied_vol — filter to rows where it's present,
    # so both models are compared on an identical, fair set of rows.
    df = df.dropna(subset=["implied_vol", "delta_C", "delta_F"]).copy()
    print(f"  {before} rows in test set, {len(df)} usable for comparison "
          f"(require implied_vol, delta_C, delta_F all present)")

    # NN predictions
    X = torch.tensor(
        ((df[FEATURES] - feature_mean) / feature_std).values, dtype=torch.float32
    )
    with torch.no_grad():
        nn_hedge_ratio = model(X).numpy()

    # Black-76 analytical delta
    bs_hedge_ratio = df.apply(
        lambda row: black76_delta(
            F=row["futures_settle"], K=row["strike_price"], T=row["ttm_years"],
            r=row["rate"], sigma=row["implied_vol"], option_type=option_type,
        ),
        axis=1,
    ).values

    delta_C = df["delta_C"].values
    delta_F = df["delta_F"].values

    nn_hedging_error_sq = (delta_C - nn_hedge_ratio * delta_F) ** 2
    bs_hedging_error_sq = (delta_C - bs_hedge_ratio * delta_F) ** 2

    mshe_nn = np.nanmean(nn_hedging_error_sq)
    mshe_bs = np.nanmean(bs_hedging_error_sq)

    print(f"\n  MSHE (neural network):  {mshe_nn:.4f}")
    print(f"  MSHE (Black-76):        {mshe_bs:.4f}")
    if mshe_nn < mshe_bs:
        improvement = (1 - mshe_nn / mshe_bs) * 100
        print(f"  -> NN reduces hedging error by {improvement:.1f}% relative to Black-76")
    else:
        worsening = (mshe_nn / mshe_bs - 1) * 100
        print(f"  -> NN INCREASES hedging error by {worsening:.1f}% relative to Black-76")

    # Diagnostic: does performance depend on the gap length between matched
    # observations? Training windows had a much longer typical gap (median
    # 7 days) than this test window (median 1 day for in-range labels) — if
    # the NN's disadvantage shrinks or reverses on longer-gap test rows,
    # that's evidence of a train/test horizon mismatch rather than a
    # fundamentally worse model.
    print(f"\n  --- By gap length (train/test horizon-mismatch check) ---")
    for gap_label, gap_filter in [
        ("gap = 1 day", df["gap_days"] == 1),
        ("gap 2-4 days", (df["gap_days"] >= 2) & (df["gap_days"] <= 4)),
        ("gap >= 5 days", df["gap_days"] >= 5),
    ]:
        n = gap_filter.sum()
        if n < 20:
            print(f"  {gap_label}: only {n} rows, skipping (too few to be meaningful)")
            continue
        mshe_nn_sub = np.nanmean(nn_hedging_error_sq[gap_filter.values])
        mshe_bs_sub = np.nanmean(bs_hedging_error_sq[gap_filter.values])
        print(f"  {gap_label} (n={n}): NN = {mshe_nn_sub:.4f} | "
              f"Black-76 = {mshe_bs_sub:.4f} | "
              f"{'NN better' if mshe_nn_sub < mshe_bs_sub else 'Black-76 better'}")

    return mshe_nn, mshe_bs


if __name__ == "__main__":
    results = {}
    for option_type, label_name in [("C", "calls"), ("P", "puts")]:
        results[label_name] = evaluate_one_type(option_type, label_name)

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for label_name, (mshe_nn, mshe_bs) in results.items():
        print(f"{label_name}: NN MSHE = {mshe_nn:.4f} | Black-76 MSHE = {mshe_bs:.4f}")