"""
features/feature_builder.py

Compute 10 technical indicators from raw OHLCV data with zero look-ahead bias
and save the result to data/aapl_features.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "log_return",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "wma_20",
    "bb_upper",
    "bb_lower",
    "volatility_10",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FEATURES_CSV = _PROJECT_ROOT / "data" / "aapl_features.csv"
_ROLLING_CSV = _PROJECT_ROOT / "data" / "aapl_rolling.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 10 features from a raw OHLCV DataFrame.

    Features computed (all using only data up to time t — zero look-ahead bias):
      log_return, sma_10, sma_20, sma_50, ema_12, ema_26,
      wma_20, bb_upper, bb_lower, volatility_10

    Drops NaN rows from warm-up period (first ~50 rows).
    Saves result to data/aapl_features.csv.

    Args:
        raw_df: DataFrame with at least a 'Close' column and a DatetimeIndex.

    Returns:
        pd.DataFrame: Feature DataFrame with exactly 10 columns in the order
                      defined by FEATURE_COLUMNS.

    Raises:
        ValueError: If input has fewer than 50 rows.
    """
    if len(raw_df) < 50:
        raise ValueError(
            f"Insufficient data: need at least 50 rows, got {len(raw_df)}"
        )

    close = raw_df["Close"]

    # --- Compute all 10 features ---
    log_return = np.log(close / close.shift(1))

    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    # Weighted moving average: weights 1..20, sum of weights = 210
    wma_20 = close.rolling(20).apply(
        lambda x: np.dot(x, np.arange(1, 21)) / 210, raw=True
    )

    rolling_std_20 = close.rolling(20).std()
    bb_upper = sma_20 + 2 * rolling_std_20
    bb_lower = sma_20 - 2 * rolling_std_20

    volatility_10 = log_return.rolling(10).std()

    # --- Assemble in the required column order ---
    features = pd.DataFrame(
        {
            "log_return": log_return,
            "sma_10": sma_10,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "ema_12": ema_12,
            "ema_26": ema_26,
            "wma_20": wma_20,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "volatility_10": volatility_10,
        },
        index=raw_df.index,
    )[FEATURE_COLUMNS]  # enforce exact column order

    # --- Drop warm-up NaN rows ---
    features = features.dropna()

    # --- Persist to CSV ---
    _FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(_FEATURES_CSV, index=True)

    return features


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Read data/aapl_rolling.csv and build features, saving to data/aapl_features.csv.

    Raises:
        FileNotFoundError: If data/aapl_rolling.csv does not exist.
    """
    if not _ROLLING_CSV.exists():
        raise FileNotFoundError(
            f"Rolling data file not found: {_ROLLING_CSV}. "
            "Run 'python -m data.live_fetcher' first."
        )

    raw_df = pd.read_csv(_ROLLING_CSV, index_col=0, parse_dates=True)
    feature_df = build_features(raw_df)
    print(
        f"Features built: {len(feature_df)} rows × {len(feature_df.columns)} columns "
        f"saved to {_FEATURES_CSV}"
    )


if __name__ == "__main__":
    main()
