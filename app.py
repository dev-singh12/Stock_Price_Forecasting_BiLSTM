import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Stock Price Forecasting",
    layout="wide"
)

st.title("📈 Stock Price Forecasting using Bidirectional LSTM")
st.write(
    "Short-term stock price forecasting using a Bidirectional LSTM model "
    "with TradingView-style technical indicators."
)

@st.cache_resource
def load_trained_model():
    return load_model("bilstm_stock_model.keras")

model = load_trained_model()

ticker = st.text_input("Stock Ticker Symbol", "AAPL")
forecast_days = st.slider("Forecast Horizon (Days)", 1, 14, 7)

@st.cache_data
def fetch_data(symbol):
    df = yf.download(symbol, start="2010-01-01")
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

if st.button("Generate Forecast"):

    try:
        data = fetch_data(ticker)

        data['SMA20'] = data['Close'].rolling(20).mean()
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

        weights = np.arange(1, 21)
        data['WMA20'] = data['Close'].rolling(20).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

        data['STD20'] = data['Close'].rolling(20).std()
        data['Upper_Band'] = data['SMA20'] + 2 * data['STD20']
        data['Lower_Band'] = data['SMA20'] - 2 * data['STD20']
        data.dropna(inplace=True)

        returns = data[['Close']].pct_change()
        returns.columns = ['Return']

        features = pd.concat(
            [
                returns,
                data[['SMA20', 'EMA20', 'WMA20', 'Upper_Band', 'Lower_Band']]
            ],
            axis=1
        )

        features.dropna(inplace=True)

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        time_step = 100
        window = scaled_features[-time_step:]
        window = window.reshape(1, time_step, scaled_features.shape[1])

        future_scaled_returns = []

        for _ in range(forecast_days):
            pred = model.predict(window, verbose=0)[0][0]
            future_scaled_returns.append(pred)

            next_row = window[0, -1, :].copy()
            next_row[0] = pred
            window = np.append(
                window[:, 1:, :],
                [[next_row]],
                axis=1
            )

        dummy = np.zeros((len(future_scaled_returns), scaled_features.shape[1]))
        dummy[:, 0] = future_scaled_returns
        predicted_returns = scaler.inverse_transform(dummy)[:, 0]

        predicted_returns = np.clip(predicted_returns, -0.05, 0.05)

        last_price = data['Close'].iloc[-1]
        forecast_prices = []

        for r in predicted_returns:
            last_price *= (1 + r)
            forecast_prices.append(last_price)

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(data['Close'].values[-150:], label="Closing Price", linewidth=2)
        ax.plot(data['SMA20'].values[-150:], label="SMA 20", linestyle="--")
        ax.plot(data['EMA20'].values[-150:], label="EMA 20", linestyle="--")
        ax.plot(data['WMA20'].values[-150:], label="WMA 20", linestyle="--")

        ax.fill_between(
            range(len(data['Upper_Band'].values[-150:])),
            data['Upper_Band'].values[-150:],
            data['Lower_Band'].values[-150:],
            alpha=0.2,
            label="Bollinger Bands"
        )

        forecast_index = range(149, 149 + forecast_days + 1)
        ax.plot(
            forecast_index[1:],
            forecast_prices,
            marker="o",
            color="red",
            linewidth=2,
            label="Forecast"
        )

        ax.axvline(149, linestyle=":", color="red")

        ax.set_title(f"{ticker} — Short-Term Price Forecast")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        st.success("Forecast generated successfully.")

    except Exception as e:
        st.error("An error occurred while generating the forecast.")
        st.exception(e)

