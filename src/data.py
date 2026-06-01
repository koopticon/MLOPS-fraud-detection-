"""Data loading, splitting, and a synthetic fallback dataset.

If `data/creditcard.csv` (the real ULB Kaggle dataset) is present it is used
directly. If not, a synthetic dataset with the *same schema and class
imbalance* is generated so the whole pipeline runs end-to-end out of the box.
Swap in the real CSV and everything downstream works unchanged.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def make_synthetic(n_rows: int = 60_000, fraud_rate: float = 0.003,
                   seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Generate a fraud-like dataset matching the ULB schema.

    Frauds are made *learnable but hard* by shifting a handful of the PCA
    features and inflating transaction amounts for the positive class.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud

    # 28 PCA-style features ~ standard normal for legit transactions.
    legit = rng.normal(0, 1, size=(n_legit, 28))
    fraud = rng.normal(0, 1, size=(n_fraud, 28))
    # Shift a few signal features for the fraud class (with overlap).
    for col, shift in [(1, 2.2), (3, -1.8), (10, 2.0), (13, -2.4), (16, 1.6)]:
        fraud[:, col] += shift

    X = np.vstack([legit, fraud])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    df = pd.DataFrame(X, columns=config.PCA_FEATURES)
    # Time: seconds elapsed; lets us split chronologically for monitoring.
    df["Time"] = rng.uniform(0, 172_800, size=len(df))           # ~2 days
    # Amount: log-normal, larger and noisier for frauds.
    amount = rng.lognormal(mean=3.0, sigma=1.0, size=len(df))
    amount[y == 1] *= rng.uniform(1.5, 4.0, size=int(y.sum()))
    df["Amount"] = np.round(amount, 2)
    df[config.TARGET] = y.astype(int)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_data() -> pd.DataFrame:
    """Load the real dataset if available, otherwise the synthetic one."""
    if config.DATA_PATH.exists():
        print(f"[data] Loading real dataset: {config.DATA_PATH}")
        df = pd.read_csv(config.DATA_PATH)
    else:
        print("[data] Real CSV not found -> generating synthetic dataset. "
              f"Place the Kaggle file at {config.DATA_PATH} to use real data.")
        df = make_synthetic()
        df.to_csv(config.DATA_DIR / "creditcard_synthetic.csv", index=False)
    print(f"[data] rows={len(df):,}  fraud_rate="
          f"{df[config.TARGET].mean():.4%}")
    return df


def split_xy(df: pd.DataFrame):
    """Stratified train/test split (preserves the rare fraud ratio)."""
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE,
        stratify=y, random_state=config.RANDOM_STATE,
    )


def time_ordered_batches(df: pd.DataFrame, n_batches: int = 6):
    """Yield chronological slices to simulate data arriving over time."""
    ordered = df.sort_values("Time").reset_index(drop=True)
    for i, chunk in enumerate(np.array_split(ordered, n_batches)):
        yield i, chunk


if __name__ == "__main__":
    frame = load_data()
    X_tr, X_te, y_tr, y_te = split_xy(frame)
    print(f"[data] train={len(X_tr):,}  test={len(X_te):,}  "
          f"test_frauds={int(y_te.sum())}")
