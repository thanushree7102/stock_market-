# How to run

**Requirements:** Python 3.10 to 3.12 (TensorFlow does not yet support every newer Python version).

1. Open a terminal in this folder.
2. (Recommended) create a virtual environment:

       python -m venv .venv
       .venv\Scripts\activate          (Windows)
       source .venv/bin/activate       (Mac/Linux)

3. Install packages:

       pip install -r requirements.txt

4. Start the dashboard:

       streamlit run app.py

   It opens at http://localhost:8501. Choose the model in the sidebar and click **Run forecast**.
   ARIMA and SARIMA take about a second; LSTM takes about a minute.

5. Optional: train the LSTM from the command line:

       python lstm_stockmarket.py --epochs 25

   This prints RMSE, MAE and R2 and saves `lstm_stock_model.keras` and `lstm_prediction_plot.png`.

## Troubleshooting
- `streamlit` not recognised: use `python -m streamlit run app.py`.
- TensorFlow install fails: check `python --version` (use 3.10 to 3.12). ARIMA and SARIMA still work without it.
- Blank chart: make sure `stock_data.csv` is in the same folder as `app.py`.
