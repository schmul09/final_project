"""
Trains the two feedforward hedge-ratio networks (one for calls, one for
puts) described in Methodology 3.1.

Architecture: 4 input features (moneyness, ttm_years, realized_vol, rate)
-> hidden layers with ReLU + He initialisation -> single output neuron.
Output activation: sigmoid for calls (bounds output to [0,1]); -sigmoid
for puts (bounds output to [-1,0]) — see 3.1.5 for the reasoning behind
training separate networks rather than one combined model.

Loss: Huber, computed against the empirical hedge_ratio_label (2.3.5),
not against price.

Validation split: chronological, not random — the most recent ~15% of
dates WITHIN each training-pool window (prior_shock, calm) separately,
consistent with data_preparation.py's original split logic. This keeps
validation representative of both regimes rather than accidentally
all-calm or all-shock.

Run from the project root: python src/train_model.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

FEATURES = ["moneyness", "ttm_years", "realized_vol", "rate"]
VALIDATION_FRACTION = 0.15

TRAIN_POOL_WINDOWS = ["train_prior_shock", "train_calm"]
MODEL_READY_DIR = "data/model_ready"
CHECKPOINT_DIR = "checkpoints_4features_wd0"

HIDDEN_SIZES = [32, 16]
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
PATIENCE = 15  # early stopping: stop if val loss hasn't improved in this many epochs
BATCH_SIZE = 256


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
        bounded = torch.sigmoid(raw_out)  # in [0, 1]
        if self.option_type == "P":
            bounded = -bounded  # rescale to [-1, 0] for puts
        return bounded.squeeze(-1)


def chronological_train_val_split(df: pd.DataFrame, val_fraction: float):
    """
    Same logic as data_preparation.py: within EACH window block separately,
    take the most recent val_fraction of dates as validation, so validation
    isn't accidentally all one regime.
    """
    val_frames, train_frames = [], []
    for window_name in TRAIN_POOL_WINDOWS:
        block = df[df["window"] == window_name]
        unique_dates = sorted(block["trade_date"].unique())
        n_val_dates = max(1, int(len(unique_dates) * val_fraction))
        val_dates = set(unique_dates[-n_val_dates:])
        val_frames.append(block[block["trade_date"].isin(val_dates)])
        train_frames.append(block[~block["trade_date"].isin(val_dates)])
    return pd.concat(train_frames, ignore_index=True), pd.concat(val_frames, ignore_index=True)


def train_one_network(option_type: str, label_name: str):
    print(f"\n{'='*60}\nTraining {label_name} network (option_type={option_type})\n{'='*60}")

    # Load and combine the training-pool windows for this option type
    pool_frames = []
    for window in TRAIN_POOL_WINDOWS:
        df = pd.read_csv(f"{MODEL_READY_DIR}/{window}_{label_name}.csv", parse_dates=["trade_date"])
        pool_frames.append(df)
    pool = pd.concat(pool_frames, ignore_index=True)

    train_df, val_df = chronological_train_val_split(pool, VALIDATION_FRACTION)
    print(f"  Train: {len(train_df)} rows | Validation: {len(val_df)} rows")

    # Standardise features — fit on TRAIN ONLY, apply the same transform to val.
    # (Fitting on val/test data would leak information about their
    # distribution into preprocessing — the same leakage principle already
    # applied to the train/val/test split itself.)
    feature_mean = train_df[FEATURES].mean()
    feature_std = train_df[FEATURES].std()

    def standardize(df):
        return (df[FEATURES] - feature_mean) / feature_std

    X_train = torch.tensor(standardize(train_df).values, dtype=torch.float32)
    y_train = torch.tensor(train_df["hedge_ratio_label"].values, dtype=torch.float32)
    X_val = torch.tensor(standardize(val_df).values, dtype=torch.float32)
    y_val = torch.tensor(val_df["hedge_ratio_label"].values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    model = HedgeRatioNet(n_features=len(FEATURES), hidden_sizes=HIDDEN_SIZES, option_type=option_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    loss_fn = nn.HuberLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_train_count = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
            total_train_count += xb.size(0)
        train_loss = total_train_loss / total_train_count

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: train Huber loss = {train_loss:.5f} | "
                  f"val Huber loss = {val_loss:.5f} (best val = {best_val_loss:.5f})")

        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)

    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = f"{CHECKPOINT_DIR}/{label_name}_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean.to_dict(),
        "feature_std": feature_std.to_dict(),
        "features": FEATURES,
        "option_type": option_type,
        "hidden_sizes": HIDDEN_SIZES,
    }, checkpoint_path)
    print(f"  Saved {checkpoint_path} (best val Huber loss: {best_val_loss:.5f})")

    return model, feature_mean, feature_std


if __name__ == "__main__":
    call_model, call_mean, call_std = train_one_network("C", "calls")
    put_model, put_mean, put_std = train_one_network("P", "puts")

    print("\nDone. Both networks trained and saved to checkpoints/.")
    print("Next step: evaluate on the held-out test_current_shock window "
          "and compare hedging performance against Black-76 (Section 3.3).")