import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings("ignore")

# ---------------------- Load Data --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('1_min_SPY_2008-2021.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# ---------------------- LSTM Helpers --------------------------
def reshape_for_lstm(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(30, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def forecast_lstm(y_series, steps=100, n_steps=20):
    data = y_series.to_numpy()
    X_train, y_train = reshape_for_lstm(data, n_steps)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = build_lstm_model((n_steps, 1))
    model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=0)

    forecast_input = data[-n_steps:].reshape(1, n_steps, 1)
    forecast = []
    for _ in range(steps):
        pred = model.predict(forecast_input, verbose=0)[0][0]
        forecast.append(pred)
        forecast_input = np.append(forecast_input[:, 1:, :], [[[pred]]], axis=1)

    return pd.Series(forecast)

# ---------------------- Classical Forecast Models --------------------------
def forecast_model(model_type, y, steps=100):
    if model_type == 'ARIMA':
        model = ARIMA(y, order=(2, 1, 2)).fit()
        forecast_obj = model.get_forecast(steps=steps)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()
        return forecast, conf_int
    elif model_type == 'SARIMA':
        model = SARIMAX(y, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        forecast_obj = model.get_forecast(steps=steps)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()
        return forecast, conf_int
    elif model_type == 'LSTM':
        forecast = forecast_lstm(y, steps=steps)
        return forecast, None
    else:
        raise ValueError("Unsupported model type")

# ---------------------- Emotion Detection --------------------------
def detect_emotion(forecast):
    trend = forecast.diff().fillna(0)
    up = trend[trend > 0].count()
    down = trend[trend < 0].count()
    if up > down:
        return "📈 Bullish", "The stock shows more upward trends"
    elif down > up:
        return "📉 Bearish", "The stock shows more downward trends"
    else:
        return "😐 Neutral", "The trend is stable with no major change"

# ---------------------- Streamlit UI --------------------------
st.set_page_config(page_title="Stock Forecasting App", layout="wide")
st.title("📊 Stock Forecasting App with Emotion Detection (ARIMA, SARIMA, LSTM)")

df = load_data()
stock_options = df['symbol'].unique() if 'symbol' in df.columns else ['SPY']
stock = st.selectbox("Select Stock", stock_options)
date_range = st.date_input("Select Date Range", [df.index.min().date(), df.index.max().date()])
model_type = st.selectbox("Choose Forecasting Model", ['ARIMA', 'SARIMA', 'LSTM'])

n_days = st.slider("Forecast Steps (minutes)", min_value=10, max_value=200, value=60)

if st.button("Run Forecast"):
    with st.spinner("⏳ Forecasting in progress..."):
        filtered_df = df[df.index.date >= date_range[0]]
        filtered_df = filtered_df[filtered_df.index.date <= date_range[1]]
        y = filtered_df['close']

        if len(y) < 200:
            st.warning("Not enough data to forecast.")
        else:
            forecast, conf_int = forecast_model(model_type, y, steps=n_days)

            # Plotting
            future_index = pd.date_range(start=y.index[-1], periods=n_days + 1, freq='T')[1:]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y.index, y=y, name="Actual", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=future_index, y=forecast, name="Forecast", line=dict(color='green')))

            if conf_int is not None:
                fig.add_trace(go.Scatter(x=future_index, y=conf_int.iloc[:, 0], name="Lower Bound", line=dict(color='lightgreen', dash='dot')))
                fig.add_trace(go.Scatter(x=future_index, y=conf_int.iloc[:, 1], name="Upper Bound", line=dict(color='lightgreen', dash='dot')))

            fig.update_layout(title="📉 Forecasted Stock Prices", xaxis_title="Time", yaxis_title="Price", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Emotion Detection
            emotion, message = detect_emotion(forecast)
            st.subheader(f"Emotion: {emotion}")
            st.caption(message)

            # Download Forecast Data
            forecast_df = pd.DataFrame({
                'timestamp': future_index,
                'forecasted_close': forecast.values
            })

            if conf_int is not None:
                forecast_df['lower_bound'] = conf_int.iloc[:, 0].values
                forecast_df['upper_bound'] = conf_int.iloc[:, 1].values

            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forecast CSV", data=csv, file_name=f"{stock}_forecast.csv", mime='text/csv')