# Architecture Notes — AAPL Stock Forecasting System

## What this system does

Short-term AAPL price trend estimation using a Bidirectional LSTM trained on daily
OHLCV data and technical indicators. The goal is directional trend estimation over a
7-trading-day horizon, not exact price prediction. Stock prices are noisy, driven by
news and macro factors no model can see — so the system is designed to be honest about
that from the ground up.

---

## Model architecture

The core model is a two-layer BiLSTM followed by a Bahdanau-style attention layer and
a single Dense output unit.

```
Input (100 timesteps × 10 features)
  → Bidirectional LSTM (64 units, return_sequences=True)
  → Bidirectional LSTM (64 units, return_sequences=True)
  → Bahdanau Attention  (learns which timesteps matter most)
  → Dense (1)           (predicted log return for next trading day)
```

The attention layer is lightweight — one learned weight matrix and a context vector.
It lets the model up-weight recent volatility spikes or trend breaks within the 100-day
window without adding transformer-level complexity.

The model predicts **log returns** (`ln(P_t / P_{t-1})`), not raw prices. Log returns
are stationary and additive, which makes them better targets for LSTM training than
raw price levels, which are non-stationary and scale-dependent.

---

## Input features (10 total, AAPL-only)

All features are computed from AAPL OHLCV data only. No external correlation features
(e.g. GOOG) are used — they add noise and create dependency on a second data feed.

| Feature | Description |
|---------|-------------|
| `log_return` | Daily log return — the prediction target |
| `sma_10`, `sma_20`, `sma_50` | Simple moving averages at 3 horizons |
| `ema_12`, `ema_26` | Exponential moving averages (MACD components) |
| `wma_20` | Linearly weighted moving average (recent days weighted higher) |
| `bb_upper`, `bb_lower` | Bollinger Bands — 20-day SMA ± 2 standard deviations |
| `volatility_10` | 10-day rolling standard deviation of log returns |

All indicators use only data available at or before time `t`. No look-ahead.

---

## Sequence window and training split

The model sees a rolling window of 100 consecutive trading days as input. Sequences
are built with a sliding window: `X[i] = features[i:i+100]`, `y[i] = log_return[i+100]`.

Train/validation split is strictly chronological — the most recent 20% of sequences
form the validation set. No shuffling at any point. This is non-negotiable for
time-series: random splits leak future information into training.

The scaler (RobustScaler, not MinMaxScaler — stock data has outliers) is fit only on
the training portion. Fitting on the full dataset would leak validation-period statistics
into the scaler, which is a subtle but real form of data leakage.

---

## Recursive 7-day forecast

At inference time, the model generates a 7-step forward forecast recursively:

1. Take the last 100 days of scaled features as the initial window.
2. Predict the next log return (scaled output from model).
3. Inverse-transform to recover the raw log return.
4. Clip to ±5% per step (`±ln(1.05)`) — a single-day AAPL move beyond ±5% is a
   black swan event; anything larger is model drift, not signal.
5. Convert to price: `P_{t+1} = P_t × exp(log_return)`.
6. Re-scale the clipped log return back to scaled space.
7. Carry forward all other features (SMAs, EMAs, etc.) from the last known row —
   recomputing them recursively would require price history that doesn't exist yet.
8. Shift the window forward and repeat for 7 steps.

Each forecast point gets a ±1.5% uncertainty band. This is a fixed heuristic, not a
statistically derived confidence interval — it's honest about that in the UI.

---

## Scaling strategy

RobustScaler is used instead of MinMaxScaler. AAPL's price history includes earnings
surprises, flash crashes, and COVID-era volatility — all of which produce outliers that
distort MinMaxScaler's range. RobustScaler uses median and IQR, making it more stable
under these conditions.

---

## Retraining pipeline

The model is retrained on a rolling 3-year window of AAPL data. Each retrain:
- Refreshes features from the latest rolling CSV
- Fits a new scaler on the training split only
- Trains from scratch (not fine-tuning) with early stopping (patience=10 on val_loss)
- Saves the best checkpoint as `models/model_v{YYYYMMDD}.keras`
- Keeps the last 3 model versions on disk; older ones are deleted automatically

Retraining is triggered manually via the dashboard or by an external cron job
(`python -m training.retrain_pipeline`). It does not run automatically on app startup.

---

## Forecast evaluation

After forecast dates pass, the system fetches actual AAPL closing prices and computes:

- **MAE** (mean absolute error in USD)
- **RMSE** (root mean squared error in USD)
- **Directional accuracy** — the fraction of days where the predicted direction
  (up or down from the last known price) matched the actual direction

Directional accuracy above ~55% over 30+ forecasts suggests the model has learned
a real signal. Below 50% is no better than a coin flip. The dashboard is explicit
about this threshold.

Partial horizons (fewer than 7 actual prices available yet) are handled gracefully —
metrics are computed on whatever days are available.

---

## Why exact price prediction is unrealistic

Stock prices incorporate all publicly available information (efficient market hypothesis)
plus noise from order flow, sentiment, and macro events that no price-based model can
observe. A model trained only on historical OHLCV data cannot predict earnings surprises,
Fed announcements, or geopolitical events. The system is designed to estimate short-term
trend direction and magnitude, not to produce tradeable price targets.
