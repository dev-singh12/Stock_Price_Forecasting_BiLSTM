import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Stock Market Forecasting",
    layout="wide"
)

st.title("📈 Stock Market Forecasting using Bidirectional LSTM")
st.write(
    "Short-term (7 trading days) stock price forecasting using a Bidirectional LSTM "
    "model with technical indicators and correlated stock analysis."
)

@st.cache_resource
def load_trained_model():
    return load_model("bilstm_stock_model.keras")

model = load_trained_model()

ticker = st.text_input("Primary Stock Ticker", "AAPL")
forecast_days = 7
st.write("Forecast Horizon: **7 Trading Days**")

@st.cache_data
def fetch_stock(symbol):
    df = yf.download(symbol, start="2010-01-01", progress=False)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

if st.button("Generate Forecast"):

    try:
        aapl = fetch_stock(ticker)
        goog = fetch_stock("GOOG").rename(columns={"Close": "GOOG_Close"})

        data = aapl.merge(
            goog,
            left_index=True,
            right_index=True,
            how="inner"
        )

        if len(data) < 150:
            st.error("Not enough historical data to generate forecast.")
            st.stop()

        data['SMA20'] = data['Close'].rolling(20).mean()
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

        weights = np.arange(1, 21)
        data['WMA20'] = data['Close'].rolling(20).apply(
            lambda x: np.dot(x, weights) / weights.sum(),
            raw=True
        )

        data['STD20'] = data['Close'].rolling(20).std()
        data['Upper_Band'] = data['SMA20'] + 2 * data['STD20']
        data['Lower_Band'] = data['SMA20'] - 2 * data['STD20']

        data['AAPL_Return'] = data['Close'].pct_change()
        data['GOOG_Return'] = data['GOOG_Close'].pct_change()

        features = data[
            [
                'AAPL_Return',
                'GOOG_Return',
                'SMA20',
                'EMA20',
                'WMA20',
                'Upper_Band',
                'Lower_Band'
            ]
        ].dropna()

        features.columns = [str(col) for col in features.columns]

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        time_step = 100
        window = scaled_features[-time_step:].reshape(
            1, time_step, scaled_features.shape[1]
        )

        future_scaled_returns = []

        for _ in range(forecast_days):
            pred = model.predict(window, verbose=0)[0][0]
            future_scaled_returns.append(pred)

            next_row = window[0, -1].copy()
            next_row[0] = pred
            window = np.concatenate(
                [window[:, 1:, :], next_row.reshape(1, 1, -1)],
                axis=1
            )

        dummy = np.zeros((forecast_days, scaled_features.shape[1]))
        dummy[:, 0] = future_scaled_returns
        predicted_returns = scaler.inverse_transform(dummy)[:, 0]
        predicted_returns = np.clip(predicted_returns, -0.05, 0.05)

        last_price = data['Close'].iloc[-1]
        forecast_prices = []

        for r in predicted_returns:
            last_price *= (1 + r)
            forecast_prices.append(last_price)

        recent = data.iloc[-150:]

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(recent.index, recent['Close'], label="Closing Price", linewidth=2)
        ax.plot(recent.index, recent['SMA20'], label="SMA 20", linestyle="--")
        ax.plot(recent.index, recent['EMA20'], label="EMA 20", linestyle="--")
        ax.plot(recent.index, recent['WMA20'], label="WMA 20", linestyle="--")

        ax.fill_between(
            recent.index,
            recent['Upper_Band'],
            recent['Lower_Band'],
            alpha=0.2,
            label="Bollinger Bands"
        )

        future_index = pd.date_range(
            start=recent.index[-1],
            periods=forecast_days + 1,
            freq="B"
        )[1:]

        ax.plot(
            future_index,
            forecast_prices,
            marker="o",
            color="red",
            linewidth=2,
            label="7-Day Forecast"
        )

        ax.axvline(recent.index[-1], linestyle=":", color="red")

        ax.set_title(f"{ticker} — 7 Day Price Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        st.success("7-day forecast generated successfully.")

    except Exception as e:
        st.error("An error occurred while generating the forecast.")
        st.exception(e)





