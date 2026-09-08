"""
Data exploration figures for the dissertation Data section.

Generates:
  1. Futures settlement price over time, across all three windows —
     visually anchors the calm-vs-shock narrative.
  2. Daily average implied volatility over time, across all three windows —
     the core visual evidence for "volatility regime shifts during shocks,"
     which is the whole premise of using a NN over Black-76.
  3. Volatility smile comparison: implied vol vs. moneyness, on one calm day
     vs. one shock day — directly illustrates why a constant-volatility
     model (Black-76) is a poor fit exactly when it matters most.
  4. Moneyness distribution — shows the option chain's structural coverage.

Run from the project root: python src/generate_figures.py
Figures are saved to figures/ as PNG files, ready to insert into the
dissertation (diagrams don't count toward the word limit).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

FILES = {
    "train_prior_shock": "data/raw/brent_train_prior_shock.csv",
    "train_calm": "data/raw/brent_train_calm.csv",
    "test_current_shock": "data/raw/brent_test_current_shock.csv",
}

WINDOW_LABELS = {
    "train_prior_shock": "Prior shock (Sep–Dec 2025)",
    "train_calm": "Calm baseline (Jan–Feb 2026)",
    "test_current_shock": "Current shock (May–Jun 2026)",
}

WINDOW_COLORS = {
    "train_prior_shock": "#D62728",  # red — shock
    "train_calm": "#2CA02C",         # green — calm
    "test_current_shock": "#D62728", # red — shock (same family as prior shock)
}

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Load all data
# ----------------------------------------------------------------------

dfs = {}
for window_name, path in FILES.items():
    df = pd.read_csv(path, parse_dates=["trade_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    dfs[window_name] = df

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ----------------------------------------------------------------------
# Figure 1 — Futures settlement price over time, all windows
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
for window_name, df in dfs.items():
    daily = df.groupby("trade_date")["futures_settle"].mean().sort_index()
    ax.plot(daily.index, daily.values, label=WINDOW_LABELS[window_name],
            color=WINDOW_COLORS[window_name], linewidth=1.5)

ax.set_xlabel("Date")
ax.set_ylabel("Brent futures settlement price (USD)")
ax.set_title("Brent futures price across sample windows")
ax.legend(loc="best", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig1_futures_price_over_time.png")
plt.close(fig)
print(f"Saved {OUTPUT_DIR}/fig1_futures_price_over_time.png")

# ----------------------------------------------------------------------
# Figure 2 — Daily average implied volatility over time, all windows
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
for window_name, df in dfs.items():
    daily_iv = df.dropna(subset=["implied_vol"]).groupby("trade_date")["implied_vol"].mean().sort_index()
    ax.plot(daily_iv.index, daily_iv.values, label=WINDOW_LABELS[window_name],
            color=WINDOW_COLORS[window_name], linewidth=1.5, marker="o", markersize=3)

ax.set_xlabel("Date")
ax.set_ylabel("Mean implied volatility (annualised)")
ax.set_title("Implied volatility across sample windows")
ax.legend(loc="best", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig2_implied_vol_over_time.png")
plt.close(fig)
print(f"Saved {OUTPUT_DIR}/fig2_implied_vol_over_time.png")

# ----------------------------------------------------------------------
# Figure 3 — Volatility smile: calm day vs. shock day
# ----------------------------------------------------------------------

calm_df = dfs["train_calm"].dropna(subset=["implied_vol"])
shock_df = dfs["test_current_shock"].dropna(subset=["implied_vol"])

# Pick the date in each window with the most observations (most complete
# smile), rather than an arbitrary fixed date.
calm_date = calm_df["trade_date"].value_counts().idxmax()
shock_date = shock_df["trade_date"].value_counts().idxmax()

calm_day = calm_df[calm_df["trade_date"] == calm_date]
shock_day = shock_df[shock_df["trade_date"] == shock_date]

# moneyness column may not exist yet if you're plotting straight from raw
# CSVs rather than after data_preparation.py — compute it here to be safe.
for d in (calm_day, shock_day):
    if "moneyness" not in d.columns:
        d["moneyness"] = d["strike_price"] / d["futures_settle"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(calm_day["moneyness"], calm_day["implied_vol"],
           label=f"Calm day ({calm_date.date()})", color="#2CA02C", s=15, alpha=0.6)
ax.scatter(shock_day["moneyness"], shock_day["implied_vol"],
           label=f"Shock day ({shock_date.date()})", color="#D62728", s=15, alpha=0.6)

ax.set_xlabel("Moneyness (strike / futures price)")
ax.set_ylabel("Implied volatility")
ax.set_title("Volatility smile: calm vs. shock period")
ax.set_xlim(0.5, 1.5)
ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig3_vol_smile_calm_vs_shock.png")
plt.close(fig)
print(f"Saved {OUTPUT_DIR}/fig3_vol_smile_calm_vs_shock.png")

# ----------------------------------------------------------------------
# Figure 4 — Moneyness distribution (option chain coverage)
# ----------------------------------------------------------------------

all_data = pd.concat(dfs.values(), ignore_index=True)
all_data["moneyness"] = all_data["strike_price"] / all_data["futures_settle"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(all_data["moneyness"].clip(0.3, 2.5), bins=60, color="#1F77B4", alpha=0.8)
ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, label="At-the-money")
ax.set_xlabel("Moneyness (strike / futures price)")
ax.set_ylabel("Number of observations")
ax.set_title("Distribution of option moneyness across the full sample")
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig4_moneyness_distribution.png")
plt.close(fig)
print(f"Saved {OUTPUT_DIR}/fig4_moneyness_distribution.png")

print("\nAll figures saved to figures/. Ready to insert into the dissertation.")
