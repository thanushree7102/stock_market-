"""Train and evaluate the LSTM on stock_data.csv (runs locally, no Colab needed).

Usage:  python lstm_stockmarket.py [--stock AABA] [--epochs 25]
Outputs: lstm_stock_model.keras, lstm_prediction_plot.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from config import SEQ_LENGTH, load_data, make_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default=None, help="ticker name in the CSV (default: first)")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    tf.random.set_seed(42)
    np.random.seed(42)

    df = load_data()
    stock = args.stock or sorted(df["name"].unique())[0]
    df = df[df["name"] == stock]
    print(f"Stock: {stock} | rows: {len(df)} | {df['date'].min().date()} to {df['date'].max().date()}")

    close = df["close"].to_numpy(dtype=float).reshape(-1, 1)
    split = int(len(close) * 0.8)

    scaler = MinMaxScaler()
    scaler.fit(close[:split])               # fit on training data only (no leakage)
    scaled = scaler.transform(close).ravel()

    X, y = make_sequences(scaled, SEQ_LENGTH)
    train_n = split - SEQ_LENGTH
    X_train, y_train, X_test, y_test = X[:train_n], y[:train_n], X[train_n:], y[train_n:]

    model = Sequential([
        Input(shape=(SEQ_LENGTH, 1)),
        LSTM(50, return_sequences=True), Dropout(0.2),
        LSTM(50), Dropout(0.2),
        Dense(25), Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, batch_size=32, epochs=args.epochs,
              validation_data=(X_test, y_test), verbose=2)

    pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    print(f"RMSE: {np.sqrt(mean_squared_error(actual, pred)):.4f}")
    print(f"MAE : {mean_absolute_error(actual, pred):.4f}")
    print(f"R2  : {r2_score(actual, pred):.4f}")

    last = scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    nxt = scaler.inverse_transform(model.predict(last, verbose=0))[0][0]
    print(f"Next predicted close: {nxt:.2f}")

    model.save("lstm_stock_model.keras")
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label="Actual", color="blue")
    plt.plot(pred, label="Predicted", color="red")
    plt.title(f"{stock}: LSTM prediction vs actual (test set)")
    plt.xlabel("Test day"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig("lstm_prediction_plot.png", dpi=120)
    print("Saved lstm_stock_model.keras and lstm_prediction_plot.png")


if __name__ == "__main__":
    main()
