import json
import math
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from pathlib import Path

FORECAST_LOG_PATH = Path("logs/forecast_log.jsonl")
EVAL_LOG_PATH = Path("logs/eval_log.jsonl")

def _compute_metrics(
    predicted: list[float],
    actual: list[float],
    last_actual_price: float,
) -> dict:
    mae_usd = float(np.mean(np.abs(np.array(predicted) - np.array(actual))))
    rmse_usd = float(math.sqrt(np.mean((np.array(predicted) - np.array(actual))**2)))
    
    # Directional accuracy: mean(sign(pred - last_actual) == sign(act - last_actual))
    pred_diff = np.array(predicted) - last_actual_price
    act_diff = np.array(actual) - last_actual_price
    dir_acc = float(np.mean(np.sign(pred_diff) == np.sign(act_diff)))

    return {
        "mae_usd": mae_usd,
        "rmse_usd": rmse_usd,
        "directional_accuracy": dir_acc
    }

def run_evaluation() -> list[dict]:
    if not FORECAST_LOG_PATH.exists():
        raise FileNotFoundError(f"{FORECAST_LOG_PATH} does not exist.")
        
    idempotency_set = set()
    if EVAL_LOG_PATH.exists():
        try:
            with open(EVAL_LOG_PATH, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        if "run_timestamp" in record:
                            idempotency_set.add(record["run_timestamp"])
        except Exception:
            pass

    new_eval_records = []
    
    with open(FORECAST_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            forecast_record = json.loads(line)
            run_ts = forecast_record.get("run_timestamp")
            
            if not run_ts or run_ts in idempotency_set:
                continue
                
            forecasts = forecast_record.get("forecast", [])
            if not forecasts:
                continue
                
            dates_str = [fc["date"] for fc in forecasts]
            predicted_prices = [fc["predicted_price"] for fc in forecasts]
            
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates_str]
            min_date = min(dates)
            max_date = max(dates)
            
            actual_df = yf.download(
                "AAPL",
                start=min_date,
                end=max_date + timedelta(days=1),
                progress=False
            )["Close"]
            
            actual_df = actual_df.reindex(dates).dropna()
            
            if actual_df.empty:
                continue
                
            available_actual_list = []
            available_predicted_list = []
            for dt_obj, dt_str, pred in zip(dates, dates_str, predicted_prices):
                if dt_obj in actual_df.index:
                    val = actual_df.loc[dt_obj]
                    if hasattr(val, 'item'):
                        val = val.item()
                    available_actual_list.append(float(val))
                    available_predicted_list.append(pred)
                    
            if not available_actual_list:
                continue
                
            metrics = _compute_metrics(
                predicted=available_predicted_list,
                actual=available_actual_list,
                last_actual_price=float(forecast_record["last_actual_price"])
            )
            
            horizon_available = len(available_actual_list)
            eval_record = {
                "run_timestamp": run_ts,
                "model_version": forecast_record.get("model_version", "unknown"),
                "horizon_available": horizon_available,
                "mae_usd": metrics["mae_usd"],
                "rmse_usd": metrics["rmse_usd"],
                "directional_accuracy": metrics["directional_accuracy"],
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "computation_complete": horizon_available == len(forecasts)
            }
            
            EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(EVAL_LOG_PATH, "a") as ef:
                ef.write(json.dumps(eval_record) + "\n")
                
            new_eval_records.append(eval_record)
            
    return new_eval_records

def main() -> None:
    try:
        records = run_evaluation()
        if not records:
            print("No new forecast runs to evaluate.")
        else:
            print(f"Evaluated {len(records)} forecast run(s):")
            for r in records:
                print(
                    f"  {r['run_timestamp'][:10]}: "
                    f"MAE=${r['mae_usd']:.2f}  "
                    f"RMSE=${r['rmse_usd']:.2f}  "
                    f"Dir={r['directional_accuracy']:.1%}  "
                    f"({r['horizon_available']}/7 days)"
                )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
