"""Streamlit dashboard: stock forecasting with ARIMA, SARIMA and LSTM."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import MIN_ROWS, load_data
from forecasting import backtest, detect_trend, run_forecast

st.set_page_config(page_title="Stock Forecasting App", layout="wide")
st.title("📊 Stock Forecasting App (ARIMA, SARIMA, LSTM)")


@st.cache_data
def get_data():
    return load_data()


@st.cache_data(show_spinner=False)
def cached_forecast(model_type, values, steps):
    return run_forecast(model_type, values, steps)


@st.cache_data(show_spinner=False)
def cached_backtest(model_type, values):
    return backtest(model_type, values)


df = get_data()

with st.sidebar:
    st.header("Settings")
    stock = st.selectbox("Select stock", sorted(df["name"].unique()))
    sdf = df[df["name"] == stock]
    min_d, max_d = sdf["date"].min().date(), sdf["date"].max().date()
    date_range = st.date_input("Date range", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)
    model_type = st.selectbox("Forecasting model", ["ARIMA", "SARIMA", "LSTM"])
    steps = st.slider("Forecast horizon (trading days)", 5, 90, 30)
    run = st.button("Run forecast", type="primary")

if len(date_range) != 2:
    st.info("Select both a start and an end date.")
    st.stop()

mask = (sdf["date"].dt.date >= date_range[0]) & (sdf["date"].dt.date <= date_range[1])
data = sdf[mask]

st.subheader(f"{stock}: closing price history")
hist = go.Figure(go.Scatter(x=data["date"], y=data["close"], name="Close",
                            line=dict(color="royalblue")))
hist.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(hist, width="stretch")

if not run:
    st.caption("Choose your settings in the sidebar and click **Run forecast**.")
    st.stop()

if len(data) < MIN_ROWS:
    st.warning(f"Need at least {MIN_ROWS} rows to forecast; the selected range has {len(data)}.")
    st.stop()

values = data["close"].to_numpy(dtype=float)

with st.spinner(f"Training {model_type} and forecasting..."):
    try:
        forecast, conf = cached_forecast(model_type, values, steps)
        metrics = cached_backtest(model_type, values)
    except ImportError:
        st.error("TensorFlow is not installed. Run `pip install -r requirements.txt`.")
        st.stop()

future_dates = pd.bdate_range(start=data["date"].iloc[-1] + pd.Timedelta(days=1),
                              periods=steps)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["date"], y=values, name="Actual",
                         line=dict(color="royalblue")))
fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast",
                         line=dict(color="green")))
if conf is not None:
    fig.add_trace(go.Scatter(x=future_dates, y=conf[:, 0], name="Lower bound",
                             line=dict(color="lightgreen", dash="dot")))
    fig.add_trace(go.Scatter(x=future_dates, y=conf[:, 1], name="Upper bound",
                             line=dict(color="lightgreen", dash="dot")))
fig.update_layout(title=f"{model_type} forecast for {stock}", template="plotly_white",
                  xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, width="stretch")

label, message = detect_trend(forecast)
c1, c2 = st.columns(2)
with c1:
    st.subheader(f"Trend: {label}")
    st.caption(message)
with c2:
    st.subheader("Accuracy (last 30 days held out)")
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{metrics['RMSE']:.2f}")
    m2.metric("MAE", f"{metrics['MAE']:.2f}")
    m3.metric("MAPE", f"{metrics['MAPE %']:.1f}%")

out = pd.DataFrame({"date": future_dates, "forecasted_close": forecast})
if conf is not None:
    out["lower_bound"], out["upper_bound"] = conf[:, 0], conf[:, 1]
st.dataframe(out, width="stretch")
st.download_button("📥 Download forecast CSV", out.to_csv(index=False).encode("utf-8"),
                   file_name=f"{stock}_{model_type}_forecast.csv", mime="text/csv")
st.caption("For learning only. Not financial advice.")
