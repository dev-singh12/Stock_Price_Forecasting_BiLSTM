# 📈 Stock Price Prediction using Bidirectional LSTM

This project implements a short-term stock price forecasting system using Bidirectional Long Short-Term Memory (BiLSTM) networks combined with technical indicators inspired by TradingView charts.

The objective of this project is to predict short-term market trends and price movements rather than exact long-term stock prices.

---

## Overview

Stock markets are time-series in nature and are influenced by trends, volatility, and temporal dependencies.  
This project uses a multivariate Bidirectional LSTM model to learn these patterns from historical stock data and generate realistic short-term forecasts.

To avoid flat or unrealistic predictions, the model is trained using return-based learning, which is commonly used in real-world financial modeling.

---

## Dataset

- Source: Yahoo Finance (`yfinance`)
- Primary Stock: Apple Inc. (AAPL)
- Related Stock: Google (GOOG) for correlation analysis
- Time Period: 2010 – Present (14+ years)
- Features Used:
  - Closing Prices
  - Technical Indicators (Moving Averages, Bollinger Bands)

---

## Methodology

1. Data collection using Yahoo Finance
2. Feature engineering with technical indicators
   - Simple Moving Average (SMA)
   - Exponential Moving Average (EMA)
   - Weighted Moving Average (WMA)
   - Bollinger Bands
3. Return-based normalization using MinMaxScaler
4. Train-test split using sliding window technique (100 timesteps)
5. Multivariate Bidirectional LSTM model training
6. Short-term recursive forecasting (7 trading days)
7. TradingView-style visualization of actual and predicted prices

---

## Model Architecture

- Bidirectional LSTM Layer (64 units, return sequences)
- Bidirectional LSTM Layer (64 units)
- Dense Output Layer
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam

---

## Results

The model shows stable performance for short-term trend prediction.  
Predictions follow realistic market movements while smoothing out daily noise, which helps reduce overfitting and flat-line behavior.

Due to the stochastic nature of stock markets, prediction accuracy decreases as the forecast horizon increases.

---

## Limitations

- Uses only historical price-based indicators
- Does not include news, macroeconomic data, or sentiment analysis
- Designed for short-term forecasting, not long-term investment decisions

---

## Future Improvements

- Incorporate volume-based indicators (VWAP, OBV)
- Extend to multi-stock portfolio forecasting
- Add confidence intervals to predictions
- Explore Transformer-based time-series models

---

## Deployment

The model is deployed as an interactive Streamlit web application that allows users to:

- Enter a stock ticker
- View technical indicators
- Visualize short-term price forecasts
