"""ARIMA, SARIMA and LSTM forecasting functions."""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import LSTM_EPOCHS, SEASONAL_PERIOD, SEQ_LENGTH, make_sequences

warnings.filterwarnings("ignore")


def forecast_arima(values: np.ndarray, steps: int):
    res = ARIMA(values, order=(2, 1, 2)).fit()
    fc = res.get_forecast(steps=steps)
    return np.asarray(fc.predicted_mean), np.asarray(fc.conf_int())


def forecast_sarima(values: np.ndarray, steps: int):
    res = SARIMAX(values, order=(1, 1, 1),
                  seasonal_order=(1, 0, 1, SEASONAL_PERIOD)).fit(disp=False, maxiter=50)
    fc = res.get_forecast(steps=steps)
    return np.asarray(fc.predicted_mean), np.asarray(fc.conf_int())


def forecast_lstm(values: np.ndarray, steps: int, epochs: int = LSTM_EPOCHS):
    # TensorFlow is imported here so ARIMA/SARIMA work even if it is missing.
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    tf.random.set_seed(42)
    np.random.seed(42)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).ravel()
    X, y = make_sequences(scaled, SEQ_LENGTH)

    model = Sequential([
        Input(shape=(SEQ_LENGTH, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    window = scaled[-SEQ_LENGTH:].tolist()
    preds = []
    for _ in range(steps):
        x = np.array(window[-SEQ_LENGTH:]).reshape(1, SEQ_LENGTH, 1)
        p = float(model.predict(x, verbose=0)[0][0])
        preds.append(p)
        window.append(p)
    out = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
    return out, None


def run_forecast(model_type: str, values: np.ndarray, steps: int):
    """Returns (forecast, conf_int or None)."""
    if model_type == "ARIMA":
        return forecast_arima(values, steps)
    if model_type == "SARIMA":
        return forecast_sarima(values, steps)
    if model_type == "LSTM":
        return forecast_lstm(values, steps)
    raise ValueError(f"Unsupported model: {model_type}")


def backtest(model_type: str, values: np.ndarray, holdout: int = 30):
    """Train on all but the last `holdout` days, score on those days."""
    train, test = values[:-holdout], values[-holdout:]
    pred, _ = run_forecast(model_type, train, holdout)
    rmse = float(np.sqrt(mean_squared_error(test, pred)))
    mae = float(mean_absolute_error(test, pred))
    mape = float(np.mean(np.abs((test - pred) / test)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE %": mape}


def detect_trend(forecast: np.ndarray):
    """Simple sentiment label from the direction of the forecast."""
    diff = np.diff(forecast)
    up, down = int((diff > 0).sum()), int((diff < 0).sum())
    if up > down:
        return "📈 Bullish", "The forecast moves up on more days than it moves down."
    if down > up:
        return "📉 Bearish", "The forecast moves down on more days than it moves up."
    return "😐 Neutral", "The forecast is flat with no clear direction."
