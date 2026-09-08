"""
Compares unweighted training against loss-weighting by label reliability,
across multiple seeds (weight_decay fixed at 1e-4 throughout, since the
earlier ablation found no meaningful effect from removing it — this run
isolates the effect of reliability-weighting specifically).

Reliability weight per row combines two signals already available in the
data (Section 2.5.2):
  1. |delta_F| relative to the MIN_ABS_DELTA_F threshold — a label built
     from a larger futures move is a more stable ratio estimate than one
     sitting just above the exclusion cutoff.
  2. gap_days — a shorter gap between matched observations means less
     time for non-delta effects (time decay, vol shifts) to contaminate
     the label, so it's a closer approximation to a true instantaneous
     delta.

Both signals are combined into a single per-row weight, capped to avoid
a small number of extreme rows dominating the loss, then used to compute
a weighted-average Huber loss instead of the plain mean.

Run from the project root: python src/run_weighting_ablation.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import norm

FEATURES = ["moneyness", "ttm_years", "realized_vol", "rate"]
VALIDATION_FRACTION = 0.15
TRAIN_POOL_WINDOWS = ["train_prior_shock", "train_calm"]
MODEL_READY_DIR = "data/model_ready"
TEST_WINDOW = "test_current_shock"

HIDDEN_SIZES = [32, 16]
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
PATIENCE = 15
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-4  # fixed — see docstring

MIN_ABS_DELTA_F = 0.5  # must match the threshold used in construct_label.py
MAX_WEIGHT_CAP = 3.0   # prevents a small number of extreme rows dominating

SEEDS = [0, 1, 2, 3, 4]
CONFIGS = ["unweighted", "reliability_weighted"]


class HedgeRatioNet(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: list, option_type: str):
        super().__init__()
        assert option_type in ("C", "P")
        self.option_type = option_type
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            linear = nn.Linear(in_size, h)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
            in_size = h
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_size, 1)
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity="relu")
        nn.init.zeros_(self.output_layer.bias)

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
    val_frames, train_frames = [], []
    for window_name in TRAIN_POOL_WINDOWS:
        block = df[df["window"] == window_name]
        unique_dates = sorted(block["trade_date"].unique())
        n_val_dates = max(1, int(len(unique_dates) * val_fraction))
        val_dates = set(unique_dates[-n_val_dates:])
        val_frames.append(block[block["trade_date"].isin(val_dates)])
        train_frames.append(block[~block["trade_date"].isin(val_dates)])
    return pd.concat(train_frames, ignore_index=True), pd.concat(val_frames, ignore_index=True)


def compute_reliability_weight(df: pd.DataFrame) -> np.ndarray:
    """
    Two components, each normalised to be ~1.0 at the 'baseline' reliability
    level, multiplied together, then capped:
      - delta_F component: |delta_F| / MIN_ABS_DELTA_F (>=1 by construction,
        since rows below the threshold were already excluded upstream)
      - gap component: 1 / gap_days (a 1-day gap gets weight 1.0, a 7-day
        gap gets weight ~0.14, etc.)
    """
    for col in ["delta_F", "gap_days"]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in model-ready data — reliability "
                f"weighting needs this column to be carried through from "
                f"construct_label.py into finalise_data.py's saved output."
            )
    deltaF_component = df["delta_F"].abs() / MIN_ABS_DELTA_F
    gap_component = 1.0 / df["gap_days"].clip(lower=1)
    weight = deltaF_component * gap_component
    weight = weight.clip(upper=MAX_WEIGHT_CAP)
    return weight.values.astype(np.float32)


def train_one_network(option_type: str, label_name: str, config: str, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    pool_frames = []
    for window in TRAIN_POOL_WINDOWS:
        df = pd.read_csv(f"{MODEL_READY_DIR}/{window}_{label_name}.csv", parse_dates=["trade_date"])
        pool_frames.append(df)
    pool = pd.concat(pool_frames, ignore_index=True)

    train_df, val_df = chronological_train_val_split(pool, VALIDATION_FRACTION)

    feature_mean = train_df[FEATURES].mean()
    feature_std = train_df[FEATURES].std()

    def standardize(df):
        return (df[FEATURES] - feature_mean) / feature_std

    X_train = torch.tensor(standardize(train_df).values, dtype=torch.float32)
    y_train = torch.tensor(train_df["hedge_ratio_label"].values, dtype=torch.float32)
    X_val = torch.tensor(standardize(val_df).values, dtype=torch.float32)
    y_val = torch.tensor(val_df["hedge_ratio_label"].values, dtype=torch.float32)

    if config == "reliability_weighted":
        w_train = torch.tensor(compute_reliability_weight(train_df), dtype=torch.float32)
    else:
        w_train = torch.ones(len(train_df), dtype=torch.float32)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train, w_train), batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )

    model = HedgeRatioNet(n_features=len(FEATURES), hidden_sizes=HIDDEN_SIZES, option_type=option_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.HuberLoss(reduction="none")  # per-sample, so we can apply weights before averaging

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None
    val_loss_fn = nn.HuberLoss()  # plain mean for validation — always unweighted, for a fair comparison

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb, wb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            per_sample_loss = loss_fn(pred, yb)
            loss = (per_sample_loss * wb).sum() / wb.sum()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = val_loss_fn(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, feature_mean, feature_std


def evaluate_one_type(model, feature_mean, feature_std, option_type: str, label_name: str):
    df = pd.read_csv(f"{MODEL_READY_DIR}/{TEST_WINDOW}_{label_name}.csv", parse_dates=["trade_date"])
    df = df.dropna(subset=["implied_vol", "delta_C", "delta_F"]).copy()

    X = torch.tensor(((df[FEATURES] - feature_mean) / feature_std).values, dtype=torch.float32)
    with torch.no_grad():
        nn_hedge_ratio = model(X).numpy()

    bs_hedge_ratio = df.apply(
        lambda row: black76_delta(
            F=row["futures_settle"], K=row["strike_price"], T=row["ttm_years"],
            r=row["rate"], sigma=row["implied_vol"], option_type=option_type,
        ),
        axis=1,
    ).values

    delta_C = df["delta_C"].values
    delta_F = df["delta_F"].values

    mshe_nn = np.nanmean((delta_C - nn_hedge_ratio * delta_F) ** 2)
    mshe_bs = np.nanmean((delta_C - bs_hedge_ratio * delta_F) ** 2)
    return mshe_nn, mshe_bs


if __name__ == "__main__":
    results = {config: {"calls": [], "puts": []} for config in CONFIGS}
    bs_results = {"calls": [], "puts": []}

    for config in CONFIGS:
        print(f"\n{'#'*60}\nConfiguration: {config}\n{'#'*60}")
        for seed in SEEDS:
            print(f"\n--- seed={seed} ---")
            for option_type, label_name in [("C", "calls"), ("P", "puts")]:
                model, fmean, fstd = train_one_network(option_type, label_name, config, seed)
                mshe_nn, mshe_bs = evaluate_one_type(model, fmean, fstd, option_type, label_name)
                results[config][label_name].append(mshe_nn)
                bs_results[label_name].append(mshe_bs)
                print(f"  {label_name}: NN MSHE = {mshe_nn:.4f} | Black-76 MSHE = {mshe_bs:.4f}")

    print(f"\n{'='*60}\nSummary (mean ± std across {len(SEEDS)} seeds)\n{'='*60}")
    for label_name in ["calls", "puts"]:
        bs_mean = np.mean(bs_results[label_name])
        print(f"\n{label_name} — Black-76 MSHE: {bs_mean:.4f} (deterministic, shown once)")
        for config in CONFIGS:
            vals = results[config][label_name]
            print(f"  {config}: NN MSHE = {np.mean(vals):.4f} ± {np.std(vals):.4f} "
                  f"(runs: {[round(v, 4) for v in vals]})")