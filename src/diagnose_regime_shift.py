"""
Point 5 diagnostic: does the network's test-set error correlate with how
'out-of-distribution' each test row is, relative to what it saw in training?

Method: for each of the four features, compute mean/std from the TRAINING
data only (never touching test). For each test row, compute a z-score per
feature (how many training-std's away from the training mean), then an
overall out-of-distribution (OOD) score = max absolute z-score across the
four features. This defines 'unusual' using only training-side information
-- it's a legitimate diagnostic, not test-set leakage, since nothing about
the threshold or definition depends on the test labels or NN performance.

Then: split test rows into OOD terciles (least/middle/most unusual) and
compare NN hedging error across the three groups. If error rises with OOD
score, that's direct evidence the network struggles specifically on
observations unlike anything in its training regime -- supporting the
'prior-shock isn't teaching what current-shock needs' hypothesis.

Run from the project root: python src/diagnose_regime_shift.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

FEATURES = ["moneyness", "ttm_years", "realized_vol", "rate"]
CHECKPOINT_DIR = "checkpoints_4features"
TEST_WINDOW = "test_current_shock"
MODEL_READY_DIR = "data/model_ready"
TRAIN_POOL_WINDOWS = ["train_prior_shock", "train_calm"]


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
    return disc * (norm.cdf(d1) - 1)


def diagnose_one_type(option_type: str, label_name: str):
    print(f"\n{'='*60}\nRegime-shift diagnostic: {label_name}\n{'='*60}")

    # Load training data ONLY to establish the reference distribution
    train_frames = []
    for window in TRAIN_POOL_WINDOWS:
        df = pd.read_csv(f"{MODEL_READY_DIR}/{window}_{label_name}.csv", parse_dates=["trade_date"])
        train_frames.append(df)
    train_df = pd.concat(train_frames, ignore_index=True)

    train_mean = train_df[FEATURES].mean()
    train_std = train_df[FEATURES].std()

    # Load and evaluate the model on the test set (same as evaluate_model.py)
    checkpoint = torch.load(f"{CHECKPOINT_DIR}/{label_name}_model.pt", weights_only=False)
    model = HedgeRatioNet(
        n_features=len(FEATURES), hidden_sizes=checkpoint["hidden_sizes"], option_type=checkpoint["option_type"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    feature_mean = pd.Series(checkpoint["feature_mean"])
    feature_std = pd.Series(checkpoint["feature_std"])

    df = pd.read_csv(f"{MODEL_READY_DIR}/{TEST_WINDOW}_{label_name}.csv", parse_dates=["trade_date"])
    df = df.dropna(subset=["implied_vol", "delta_C", "delta_F"]).copy()

    # OOD score, computed using TRAINING mean/std only
    z_scores = pd.DataFrame({
        f: (df[f] - train_mean[f]) / train_std[f] for f in FEATURES
    })
    df["ood_score"] = z_scores.abs().max(axis=1)

    # NN predictions and hedging error (same metric as evaluate_model.py)
    X = torch.tensor(((df[FEATURES] - feature_mean) / feature_std).values, dtype=torch.float32)
    with torch.no_grad():
        nn_hedge_ratio = model(X).numpy()
    bs_hedge_ratio = df.apply(
        lambda row: black76_delta(
            F=row["futures_settle"], K=row["strike_price"], T=row["ttm_years"],
            r=row["rate"], sigma=row["implied_vol"], option_type=option_type,
        ), axis=1,
    ).values
    delta_C = df["delta_C"].values
    delta_F = df["delta_F"].values
    df["nn_sq_error"] = (delta_C - nn_hedge_ratio * delta_F) ** 2
    df["bs_sq_error"] = (delta_C - bs_hedge_ratio * delta_F) ** 2

    # Split into OOD terciles and compare
    df["ood_tercile"] = pd.qcut(df["ood_score"], 3, labels=["least unusual", "middle", "most unusual"])
    print(f"\n  OOD score distribution: min={df['ood_score'].min():.2f}, "
          f"median={df['ood_score'].median():.2f}, max={df['ood_score'].max():.2f}")
    print(f"\n  {'Tercile':<15} {'n':>6} {'NN MSHE':>10} {'Black-76 MSHE':>14} {'NN/BS ratio':>12}")
    for tercile in ["least unusual", "middle", "most unusual"]:
        sub = df[df["ood_tercile"] == tercile]
        nn_mshe = sub["nn_sq_error"].mean()
        bs_mshe = sub["bs_sq_error"].mean()
        print(f"  {tercile:<15} {len(sub):>6} {nn_mshe:>10.4f} {bs_mshe:>14.4f} {nn_mshe/bs_mshe:>12.2f}")

    # Which feature is driving OOD-ness most often?
    dominant_feature = z_scores.abs().idxmax(axis=1)
    print(f"\n  Feature most often responsible for high OOD score:")
    print(f"  {dominant_feature.value_counts(normalize=True).round(3).to_dict()}")


if __name__ == "__main__":
    for option_type, label_name in [("C", "calls"), ("P", "puts")]:
        diagnose_one_type(option_type, label_name)