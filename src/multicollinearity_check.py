"""
Multicollinearity check on the candidate neural network input features.

Run from the project root: python src/check_multicollinearity.py
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

FILES = {
    "train_prior_shock": "data/raw/brent_train_prior_shock.csv",
    "train_calm": "data/raw/brent_train_calm.csv",
    "test_current_shock": "data/raw/brent_test_current_shock.csv",
}

dfs = [pd.read_csv(path, parse_dates=["trade_date"]) for path in FILES.values()]
all_data = pd.concat(dfs, ignore_index=True)

all_data["moneyness"] = all_data["strike_price"] / all_data["futures_settle"]

# 20-day rolling realised volatility per contract's underlying, as the
# candidate NN feature (same construction as intended for the model itself)
all_data = all_data.sort_values(["underlying", "trade_date"])
all_data["log_return"] = (
    all_data.groupby("underlying")["futures_settle"]
    .apply(lambda s: np.log(s / s.shift(1)))
    .reset_index(level=0, drop=True)
)
all_data["realized_vol"] = (
    all_data.groupby("underlying")["log_return"]
    .transform(lambda s: s.rolling(20).std() * np.sqrt(252))
)

# rate now comes directly from the real SOFR column already merged into
# the CSVs (see src/add_real_rate.py) — no placeholder needed anymore.

features = ["moneyness", "ttm_years", "realized_vol", "rate"]
feature_data = all_data[features].dropna()
print("--- Correlation matrix ---")
print(feature_data.corr().round(3))

print("\n--- Variance Inflation Factor (VIF) ---")
print("Rule of thumb: VIF > 5-10 suggests a feature is problematically")
print("collinear with the others.\n")
for i, feature in enumerate(features):
    vif = variance_inflation_factor(feature_data.values, i)
    print(f"{feature}: VIF = {vif:.2f}")

import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# Correlation matrix heatmap
fig, ax = plt.subplots(figsize=(6, 5))
corr = feature_data.corr()
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(features)))
ax.set_yticks(range(len(features)))
ax.set_xticklabels(features, rotation=45, ha="right")
ax.set_yticklabels(features)
for i in range(len(features)):
    for j in range(len(features)):
        ax.text(j, i, f"{corr.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
ax.set_title("Correlation matrix — NN input features")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig("figures/fig7_correlation_matrix.png", dpi=150)
plt.close(fig)
print("Saved figures/fig7_correlation_matrix.png")

# VIF bar chart
vif_values = [variance_inflation_factor(feature_data.values, i) for i in range(len(features))]
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#D62728" if v > 5 else "#2CA02C" for v in vif_values]
ax.bar(features, vif_values, color=colors)
ax.axhline(5, color="grey", linestyle="--", linewidth=0.8, label="Watch threshold (5)")
ax.axhline(10, color="black", linestyle="--", linewidth=0.8, label="Concern threshold (10)")
ax.set_ylabel("VIF")
ax.set_title("Variance Inflation Factor — NN input features")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig8_vif.png", dpi=150)
plt.close(fig)
print("Saved figures/fig8_vif.png")