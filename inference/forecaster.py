"""
PyTorch recursive 7-day forecast with scaled-space consistency.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pandas_market_calendars as mcal
from math import exp, log
from datetime import datetime, timezone

from models.model_builder import BiLSTMForecaster
from models.scaler_manager import ScalerManager
from inference.forecast_logger import log_forecast


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_forecast() -> pd.DataFrame:
    # STEP 1 — Load latest model
    model_paths = sorted(Path("models").glob("model_v*.pt"))
    if not model_paths:
        raise FileNotFoundError("No .pt files found in models directory")
    model_path = model_paths[-1]
    
    device = _get_device()
    model = BiLSTMForecaster(input_dim=10, use_attention=True)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # STEP 2 — Load latest scaler
    sm = ScalerManager()
    scaler, meta = sm.load_latest()

    # STEP 3 — Load and scale features
    features_df = pd.read_csv("data/aapl_features.csv", index_col=0, parse_dates=True)
    rolling_df  = pd.read_csv("data/aapl_rolling.csv", index_col=0, parse_dates=True)
    scaled_all  = sm.transform(features_df, scaler, meta)
    
    window     = scaled_all[-100:].copy()   # shape (100, 10), float64 numpy
    last_price = float(rolling_df["Close"].iloc[-1])
    last_date  = rolling_df.index[-1]

    # STEP 4 — Generate 7 NYSE trading dates
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=last_date + pd.Timedelta(days=1),
        end_date=last_date + pd.Timedelta(days=30),
    )
    forecast_dates = schedule.index.tolist()[:7]
    if len(forecast_dates) < 7:
        raise ValueError("Fewer than 7 trading days found in calendar")

    # STEP 5 — Recursive 7-step forecast loop
    CLIP_BOUND = log(1.05)   # ≈ 0.04879 — applied to RAW log return only
    UNCERTAINTY = 0.015      # ±1.5% band
    LOG_RETURN_IDX = 0       # column 0 in feature matrix is log_return

    results = []
    for step in range(7):

        # 5a. Model prediction (in SCALED log-return space)
        with torch.no_grad():
            x = torch.tensor(
                window.reshape(1, 100, 10),
                dtype=torch.float32
            ).to(device)
            pred_scaled = model(x).item()   # scalar float

        # 5b. Inverse-transform to RAW log return space
        #     (scaler operates on full feature rows, use dummy row pattern)
        dummy_inv = np.zeros((1, 10))
        dummy_inv[0, LOG_RETURN_IDX] = pred_scaled
        raw_log_return = scaler.inverse_transform(dummy_inv)[0][LOG_RETURN_IDX]

        # 5c. Clip RAW log return to ±5% single-day move
        #     (DO NOT clip the scaled value — that is the wrong space)
        was_clipped = abs(raw_log_return) > CLIP_BOUND
        clipped_raw = float(np.clip(raw_log_return, -CLIP_BOUND, CLIP_BOUND))
        if was_clipped:
            import warnings
            warnings.warn(
                f"Step {step+1}: raw log return {raw_log_return:.4f} "
                f"clipped to {clipped_raw:.4f}",
                RuntimeWarning,
            )

        # 5d. Convert to price
        pred_price  = last_price * exp(clipped_raw)
        lower_bound = pred_price * (1 - UNCERTAINTY)
        upper_bound = pred_price * (1 + UNCERTAINTY)

        results.append({
            "date":            forecast_dates[step],
            "predicted_price": round(pred_price,  4),
            "lower_bound":     round(lower_bound, 4),
            "upper_bound":     round(upper_bound, 4),
        })

        # 5e. Re-scale clipped raw log return BACK to scaled space
        #     for carry-forward (window must stay in scaled space)
        dummy_fwd = np.zeros((1, 10))
        dummy_fwd[0, LOG_RETURN_IDX] = clipped_raw
        re_scaled_val = scaler.transform(dummy_fwd)[0][LOG_RETURN_IDX]

        # 5f. Carry forward: copy last window row, update only log_return,
        #     shift window forward by 1
        new_row = window[-1].copy()
        new_row[LOG_RETURN_IDX] = re_scaled_val
        window = np.vstack([window[1:], new_row])
        last_price = pred_price

    # STEP 6 — Build output DataFrame and log
    forecast_df = pd.DataFrame(results)
    
    log_forecast(
        forecast_df=forecast_df,
        metadata={
            "run_timestamp":     datetime.now(timezone.utc).isoformat(),
            "model_version":     model_path.stem,
            "last_actual_date":  str(rolling_df.index[-1].date()),
            "last_actual_price": float(rolling_df["Close"].iloc[-1]),
            "disclaimer":        "Estimated trend direction only",
        },
    )
    return forecast_df


def main() -> None:
    df = generate_forecast()
    print("\n7-Day AAPL Forecast:")
    print(df.to_string(index=False))
    print("\nLogged to logs/forecast_log.jsonl")


if __name__ == "__main__":
    main()
