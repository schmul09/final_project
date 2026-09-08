import pandas as pd

FILES = {
    "train_prior_shock": "data/raw/brent_train_prior_shock.csv",
    "train_calm": "data/raw/brent_train_calm.csv",
    "test_current_shock": "data/raw/brent_test_current_shock.csv",
}

for window_name, path in FILES.items():
    df = pd.read_csv(path, parse_dates=["trade_date"])
    unique_dates = sorted(df["trade_date"].dt.date.unique())
    print(f"{window_name}: {len(unique_dates)} unique dates")
    print(f"  first: {unique_dates[0]}, last: {unique_dates[-1]}")
