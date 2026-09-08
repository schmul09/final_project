import pandas as pd

for label_name in ["calls", "puts"]:
    for window in ["train_prior_shock", "train_calm"]:
        df = pd.read_csv(f"data/model_ready/{window}_{label_name}.csv")
        print(f"\n{window} / {label_name}:")
        print(df["hedge_ratio_label"].describe())
        print(f"  Values beyond [-3, 3]: {((df['hedge_ratio_label'] < -3) | (df['hedge_ratio_label'] > 3)).sum()}")