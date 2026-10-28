# Stock Market Analysis and Forecasting

Forecast stock closing prices with **ARIMA**, **SARIMA** and an **LSTM** neural network, and explore the results in an interactive **Streamlit** dashboard.

## Features
- Interactive dashboard: pick a stock, date range, model and forecast horizon
- Forecast chart with confidence bounds (ARIMA/SARIMA)
- Bullish / bearish / neutral trend label
- Accuracy check: the last 30 days are held out and scored (RMSE, MAE, MAPE)
- Download the forecast as CSV
- Standalone LSTM training script with test-set metrics and a plot

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard |
| `forecasting.py` | ARIMA, SARIMA, LSTM models, backtest, trend label |
| `config.py` | Settings and data loading |
| `lstm_stockmarket.py` | Train and evaluate the LSTM from the command line |
| `stock_data.csv` | Daily prices (AABA, 2006-2017) |
| `*.ipynb` | EDA and model comparison notebooks (need the Kaggle SPY data / Colab) |

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```
See `HOW_TO_RUN.md` for details.

## Data format
CSV with columns `Date, Open, High, Low, Close, Volume, Name`; dates as `m/d/YYYY`. Add more rows or more tickers (different `Name` values) and they appear in the dashboard automatically.

## Disclaimer
For learning only. Not financial advice.
