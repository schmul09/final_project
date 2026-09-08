"""
Computes MSHE (same metric used in evaluate_model.py) on the VALIDATION
set — reconstructing the exact same chronological split train_model.py
used internally — so it can be directly compared against the test-window
MSHE already computed. This makes the train/test generalization gap
concrete and measurable, rather than comparing two different metrics
(Huber during training vs. MSHE at test).

Run from the project root: python src/check_generalization_gap.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

FEATURES = ["moneyness", "ttm_years", "realized_vol", "rate"]
VALIDATION_FRACTION = 0.15
TRAIN_POOL_WINDOWS = ["train_prior_shock", "train_calm"]
CHECKPOINT_DIR = "checkpoints"
MODEL_READY_DIR = "data/model_ready"


class HedgeRatioNet(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: list, option_type: str):
        super().__init__()
        self.option_type = option_type
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
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


def chronological_train_val_split(df: pd.DataFrame, val_fraction: float):
    """Identical logic to train_model.py — reconstructs the same split."""
    val_frames, train_frames = [], []
    for window_name in TRAIN_POOL_WINDOWS:
        block = df[df["window"] == window_name]
        unique_dates = sorted(block["trade_date"].unique())
        n_val_dates = max(1, int(len(unique_dates) * val_fraction))
        val_dates = set(unique_dates[-n_val_dates:])
        val_frames.append(block[block["trade_date"].isin(val_dates)])
        train_frames.append(block[~block["trade_date"].isin(val_dates)])
    return pd.concat(train_frames, ignore_index=True), pd.concat(val_frames, ignore_index=True)


def check_one_type(option_type: str, label_name: str):
    print(f"\n{'='*60}\n{label_name} (option_type={option_type})\n{'='*60}")

    checkpoint = torch.load(f"{CHECKPOINT_DIR}/{label_name}_model.pt", weights_only=False)
    model = HedgeRatioNet(
        n_features=len(FEATURES),
        hidden_sizes=checkpoint["hidden_sizes"],
        option_type=checkpoint["option_type"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_mean = pd.Series(checkpoint["feature_mean"])
    feature_std = pd.Series(checkpoint["feature_std"])

    pool_frames = []
    for window in TRAIN_POOL_WINDOWS:
        df = pd.read_csv(f"{MODEL_READY_DIR}/{window}_{label_name}.csv", parse_dates=["trade_date"])
        pool_frames.append(df)
    pool = pd.concat(pool_frames, ignore_index=True)

    _, val_df = chronological_train_val_split(pool, VALIDATION_FRACTION)

    before = len(val_df)
    val_df = val_df.dropna(subset=["implied_vol", "delta_C", "delta_F"]).copy()
    print(f"  {before} validation rows, {len(val_df)} usable for MSHE comparison")

    X = torch.tensor(
        ((val_df[FEATURES] - feature_mean) / feature_std).values, dtype=torch.float32
    )
    with torch.no_grad():
        nn_hedge_ratio = model(X).numpy()

    bs_hedge_ratio = val_df.apply(
        lambda row: black76_delta(
            F=row["futures_settle"], K=row["strike_price"], T=row["ttm_years"],
            r=row["rate"], sigma=row["implied_vol"], option_type=option_type,
        ),
        axis=1,
    ).values

    delta_C = val_df["delta_C"].values
    delta_F = val_df["delta_F"].values

    mshe_nn_val = np.nanmean((delta_C - nn_hedge_ratio * delta_F) ** 2)
    mshe_bs_val = np.nanmean((delta_C - bs_hedge_ratio * delta_F) ** 2)

    print(f"  Validation MSHE (neural network): {mshe_nn_val:.4f}")
    print(f"  Validation MSHE (Black-76):       {mshe_bs_val:.4f}")

    return mshe_nn_val, mshe_bs_val


if __name__ == "__main__":
    print("Validation-set MSHE, using the identical metric already computed "
          "on the test set in evaluate_model.py — for direct comparison.")

    for option_type, label_name in [("C", "calls"), ("P", "puts")]:
        check_one_type(option_type, label_name)

    print(f"\n{'='*60}")
    print("Compare these numbers directly against evaluate_model.py's test-set")
    print("output. A validation MSHE much lower than the test MSHE (for the")
    print("NN specifically) would indicate overfitting to the training")
    print("windows rather than genuine generalization to a new shock episode.")