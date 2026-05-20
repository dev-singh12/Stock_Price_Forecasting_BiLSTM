"""
Forecast logger for AAPL model.
"""
import json
from pathlib import Path
import pandas as pd


LOG_PATH = Path("logs/forecast_log.jsonl")


def log_forecast(forecast_df: pd.DataFrame, metadata: dict) -> None:
    """
    Append one JSON record per forecast run to logs/forecast_log.jsonl.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Format forecast_df to list of dicts
    forecast_list = []
    for _, row in forecast_df.iterrows():
        forecast_list.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "predicted_price": float(row["predicted_price"]),
            "lower_bound": float(row["lower_bound"]),
            "upper_bound": float(row["upper_bound"])
        })
        
    record = metadata.copy()
    record["forecast"] = forecast_list
    
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
