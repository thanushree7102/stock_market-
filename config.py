"""Shared settings and helpers for the stock forecasting project."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "stock_data.csv"

SEQ_LENGTH = 60        # LSTM look-back window (trading days)
LSTM_EPOCHS = 8
SEASONAL_PERIOD = 5    # one trading week
MIN_ROWS = 200


def load_data(path=DATA_FILE) -> pd.DataFrame:
    """Load the CSV, normalise column names and sort by date."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    if "name" not in df.columns:
        df["name"] = "STOCK"
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    df = df.drop_duplicates(subset=["date", "name"])
    return df.reset_index(drop=True)


def make_sequences(values: np.ndarray, seq_length: int):
    """Turn a 1-D scaled series into (X, y) windows for the LSTM."""
    X, y = [], []
    for i in range(seq_length, len(values)):
        X.append(values[i - seq_length:i])
        y.append(values[i])
    X = np.array(X).reshape(-1, seq_length, 1)
    return X, np.array(y)
