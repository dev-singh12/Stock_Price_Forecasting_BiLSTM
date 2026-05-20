"""
data/live_fetcher.py

Fetches live AAPL OHLCV data from Yahoo Finance and maintains a rolling CSV.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# Paths
_DATA_DIR = Path(__file__).parent
_CSV_PATH = _DATA_DIR / "aapl_rolling.csv"
_LOGS_DIR = _DATA_DIR.parent / "logs"
_LOG_PATH = _LOGS_DIR / "fetch_log.jsonl"

_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_MIN_ROWS = 252


def _ensure_logs_dir() -> None:
    """Create logs/ directory if it does not exist."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _append_log(record: dict) -> None:
    """Append a JSON record to logs/fetch_log.jsonl."""
    _ensure_logs_dir()
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _now_utc_iso() -> str:
    """Return current UTC time as ISO-8601 string with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns that yfinance sometimes returns."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (metric, ticker) tuples — keep only the metric level
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def fetch_and_update(rolling_window_years: int = 3) -> pd.DataFrame:
    """
    Fetch AAPL OHLCV data from Yahoo Finance and update data/aapl_rolling.csv.

    - If aapl_rolling.csv exists, performs incremental fetch from last_date + 1 day.
    - Deduplicates by date index.
    - Drops rows older than rolling_window_years from today.
    - Retains only Open, High, Low, Close, Volume columns.
    - Appends a structured log entry to logs/fetch_log.jsonl.
    - Raises ValueError if fewer than 252 rows remain after fetch.
    - Raises RuntimeError (logged) if yfinance returns empty or raises.

    Returns:
        pd.DataFrame — the updated aapl_rolling.csv contents.
    """
    today = datetime.now(timezone.utc).date()
    cutoff_date = today - timedelta(days=rolling_window_years * 365)

    # Determine start date and load existing data
    existing_df: pd.DataFrame | None = None

    if _CSV_PATH.exists():
        existing_df = pd.read_csv(_CSV_PATH, index_col=0, parse_dates=True)
        existing_df.index = pd.to_datetime(existing_df.index)
        last_date = existing_df.index.max().date()
        start_date = last_date + timedelta(days=1)
    else:
        start_date = cutoff_date

    end_date = today

    # Fetch from yfinance
    try:
        new_data = yf.download(
            "AAPL",
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        error_msg = f"yfinance raised an exception: {exc}"
        _append_log({
            "timestamp": _now_utc_iso(),
            "rows_fetched": 0,
            "status": "error",
            "error_message": error_msg,
        })
        raise RuntimeError(error_msg) from exc

    if new_data.empty:
        error_msg = "yfinance returned an empty DataFrame"
        _append_log({
            "timestamp": _now_utc_iso(),
            "rows_fetched": 0,
            "status": "error",
            "error_message": error_msg,
        })
        raise RuntimeError(error_msg)

    # Flatten MultiIndex columns if present
    new_data = _flatten_multiindex_columns(new_data)

    # Retain only OHLCV columns that are present
    available_cols = [c for c in _OHLCV_COLUMNS if c in new_data.columns]
    new_data = new_data[available_cols]

    # Ensure DatetimeIndex
    new_data.index = pd.to_datetime(new_data.index)

    rows_fetched = len(new_data)

    # Merge with existing data
    if existing_df is not None:
        # Keep only OHLCV columns from existing data too
        existing_cols = [c for c in _OHLCV_COLUMNS if c in existing_df.columns]
        existing_df = existing_df[existing_cols]
        df = pd.concat([existing_df, new_data])
    else:
        df = new_data.copy()

    # Deduplicate by date index, keeping last occurrence
    df = df[~df.index.duplicated(keep="last")]

    # Sort ascending
    df.sort_index(inplace=True)

    # Rolling window trim: drop rows older than rolling_window_years
    cutoff_ts = pd.Timestamp(cutoff_date)
    df = df[df.index >= cutoff_ts]

    # Atomic write
    tmp_path = _CSV_PATH.with_suffix(".tmp")
    df.to_csv(tmp_path)
    os.replace(tmp_path, _CSV_PATH)

    # Minimum rows check
    if len(df) < _MIN_ROWS:
        raise ValueError(
            f"Insufficient data: only {len(df)} rows after fetch, need at least {_MIN_ROWS}"
        )

    # Log success
    _append_log({
        "timestamp": _now_utc_iso(),
        "rows_fetched": rows_fetched,
        "date_range_start": str(df.index.min().date()),
        "date_range_end": str(df.index.max().date()),
        "status": "success",
    })

    return df


def main() -> None:
    df = fetch_and_update()
    print(
        f"Updated aapl_rolling.csv: {len(df)} rows, "
        f"{df.index[0].date()} to {df.index[-1].date()}"
    )


if __name__ == "__main__":
    main()
