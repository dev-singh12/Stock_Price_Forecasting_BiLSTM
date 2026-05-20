"""
Versioned RobustScaler management for AAPL feature scaling.

The scaler is fit on training data only. Using RobustScaler instead of
MinMaxScaler because AAPL data contains earnings-surprise and flash-crash
outliers that would distort MinMaxScaler's range.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler


class ScalerManager:
    """
    Manages versioned RobustScaler artifacts under the models/ directory.

    Naming convention:
        models/scaler_v{YYYYMMDD}.pkl
        models/scaler_v{YYYYMMDD}_meta.json
    """

    FEATURE_COLUMNS: list[str] = [
        "log_return", "sma_10", "sma_20", "sma_50",
        "ema_12", "ema_26", "wma_20", "bb_upper", "bb_lower", "volatility_10",
    ]

    MODELS_DIR = Path("models")

    def fit_and_save(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        version_date: str,
    ) -> RobustScaler:
        """
        Fit RobustScaler on df.iloc[:train_end_idx] ONLY.

        Saves:
            models/scaler_v{version_date}.pkl
            models/scaler_v{version_date}_meta.json

        Metadata JSON schema:
        {
          "fit_date":        "YYYYMMDD",
          "feature_columns": [...],
          "train_end_date":  "YYYY-MM-DD" (last date in training portion),
          "scaler_type":     "RobustScaler"
        }

        Args:
            df: Full feature DataFrame (all rows, not just training).
            train_end_idx: Exclusive upper row index for fitting.
            version_date: String in "YYYYMMDD" format.

        Returns:
            The fitted RobustScaler.

        Raises:
            ValueError: If df.columns don't contain all FEATURE_COLUMNS.
        """
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        missing_cols = [c for c in self.FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        train_df = df.iloc[:train_end_idx][self.FEATURE_COLUMNS]
        scaler = RobustScaler()
        scaler.fit(train_df.values)

        scaler_path = self.MODELS_DIR / f"scaler_v{version_date}.pkl"
        meta_path = self.MODELS_DIR / f"scaler_v{version_date}_meta.json"

        joblib.dump(scaler, scaler_path)

        train_end_date = str(df.index[train_end_idx - 1].date())
        meta = {
            "fit_date": version_date,
            "feature_columns": self.FEATURE_COLUMNS,
            "train_end_date": train_end_date,
            "scaler_type": "RobustScaler"
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return scaler

    def load_latest(self) -> tuple[RobustScaler, dict]:
        """
        Load the most recently dated scaler by lexicographic sort of YYYYMMDD suffix.

        Returns:
            (fitted_scaler, metadata_dict)

        Raises:
            FileNotFoundError: If no scaler_v*.pkl files exist in models/.
        """
        scaler_files = sorted(self.MODELS_DIR.glob("scaler_v*.pkl"))
        if not scaler_files:
            raise FileNotFoundError("No scaler_v*.pkl files found in models/.")
        
        latest_scaler_path = scaler_files[-1]
        latest_meta_path = self.MODELS_DIR / f"{latest_scaler_path.stem}_meta.json"

        scaler = joblib.load(latest_scaler_path)
        with open(latest_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return scaler, meta

    def transform(
        self,
        df: pd.DataFrame,
        scaler: RobustScaler,
        meta: dict,
    ) -> np.ndarray:
        """
        Transform df using the provided scaler.

        Validates that df.columns contains all columns in meta["feature_columns"]
        in the correct order before transforming.

        Returns:
            np.ndarray of shape (len(df), len(FEATURE_COLUMNS))

        Raises:
            ValueError: If column list doesn't match meta["feature_columns"].
        """
        feature_cols = meta.get("feature_columns", self.FEATURE_COLUMNS)
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        return scaler.transform(df[feature_cols].values)
