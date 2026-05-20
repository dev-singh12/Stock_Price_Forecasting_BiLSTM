# Feature 1 — Live AAPL Forecasting: Implementation Roadmap

The existing system uses a static historical dataset and a pre-trained model with no
retraining loop. This feature replaces that with a live data pipeline, periodic
retraining, and a proper forecast evaluation system — all scoped to AAPL only.

---

## Live data ingestion

`data/live_fetcher.py` fetches AAPL OHLCV data from Yahoo Finance and maintains a
rolling 3-year CSV (`data/aapl_rolling.csv`). On first run it downloads the full
3-year history; on subsequent runs it fetches only new rows since the last stored date.

Key details:
- Deduplicates by date index — safe to run multiple times per day
- Drops rows older than 3 years before writing
- Writes atomically (`os.replace`) to prevent partial-write corruption
- Logs every fetch (success or failure) to `logs/fetch_log.jsonl`
- Raises if fewer than 252 rows are available after fetch (not enough for training)

---

## Feature recomputation

`features/feature_builder.py` reads the rolling CSV and computes 10 technical
indicators, saving the result to `data/aapl_features.csv`. This runs as part of every
retrain cycle to ensure features reflect the latest data.

All indicators use only data available at or before each time step — no look-ahead.
The warm-up period (first ~50 rows) is dropped before saving.

---

## Model architecture and scaling

The model is a two-layer BiLSTM with an optional Bahdanau attention layer, trained on
100-day sliding windows of scaled features. The target is the next-day log return.

Scaling uses RobustScaler (median/IQR), fit only on the training portion of each
rolling window. The scaler is versioned alongside the model so inference always uses
the matching scaler.

---

## Retraining strategy

`training/retrain_pipeline.py` orchestrates a full retrain on the current rolling
window. It runs the feature builder, fits a new scaler, builds sequences, trains with
early stopping (patience=10), and saves the best checkpoint.

Models are versioned as `model_v{YYYYMMDD}.keras`. The pipeline keeps the last 3
versions and deletes older ones automatically. Every retrain is logged to
`logs/retrain_log.jsonl` with train/val sizes, best val loss, epochs run, and timing.

Retraining is triggered two ways:
- Manually via the "Retrain Now" button in the dashboard
- Via an external cron job: `python -m training.retrain_pipeline`

It does not run on app startup.

---

## Recursive forecasting

`inference/forecaster.py` loads the latest model and scaler, takes the last 100 days
of features as the input window, and generates a 7-step recursive forecast.

At each step: the model outputs a scaled log return → inverse-transform to raw →
clip to ±5% → convert to price → re-scale → carry forward into the next window step.

The ±5% clip prevents the model from predicting black-swan-level single-day moves.
Each forecast point gets a ±1.5% uncertainty band. All 7 forecast dates are valid
NYSE trading days (weekends and holidays skipped via `pandas_market_calendars`).

Every forecast run is logged to `logs/forecast_log.jsonl`.

---

## Evaluation

`evaluation/comparator.py` reads past forecast logs, fetches actual AAPL closing
prices for those dates, and computes MAE, RMSE, and directional accuracy per run.

Partial horizons (forecast dates that haven't passed yet) are handled — metrics are
computed on whatever actual prices are available. The comparator is idempotent: running
it multiple times won't produce duplicate evaluation records.

Results go to `logs/eval_log.jsonl`.

---

## Dashboard

`app.py` is rewritten as a 4-tab Streamlit dashboard:

- **Live Data Status** — last fetch time, latest price, data date range
- **Forecast** — Plotly chart with last 60 days of actual prices + 7-day forecast
  with shaded uncertainty band. Labeled as estimated trend, not a price target.
- **Model Status** — current model version, last retrain date, val loss.
  "Retrain Now" and "Refresh Data" buttons with spinners.
- **Evaluation History** — table and bar chart of the last 10 forecast evaluations,
  with an honest note about what directional accuracy above/below 50% means.

All tabs degrade gracefully when data files don't exist yet (early phases, fresh
install). No unhandled exceptions visible to the user.

---

## Commit structure

Each phase is independently committable and leaves the app in a runnable state:

1. Project structure + dependencies
2. `data/live_fetcher.py`
3. `features/feature_builder.py`
4. `models/` — attention, model builder, scaler manager, sequence builder
5. `training/` — retrain pipeline + logger
6. `inference/` — forecaster + forecast logger
7. `evaluation/` — comparator
8. `app.py` rewrite
