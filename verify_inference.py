import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def verify_inference():
    print("Loading artifacts...")
    try:
        model = load_model("bilstm_stock_model.keras")
        scaler = joblib.load("scaler.save")
        print("Artifacts loaded.")
    except Exception as e:
        print(f"Artifact load failed: {e}")
        return

    print("Loading data...")
    try:
        aapl = pd.read_csv("AAPL_data.csv", index_col=0, parse_dates=True)
        goog = pd.read_csv("GOOG_data.csv", index_col=0, parse_dates=True)
        print(f"Data loaded. AAPL shape: {aapl.shape}, GOOG shape: {goog.shape}")
    except Exception as e:
        print(f"Data load failed: {e}")
        return

    # Data Prep
    print("Preparing data...")
    aapl["Close"] = pd.to_numeric(aapl["Close"], errors="coerce")
    goog["Close"] = pd.to_numeric(goog["Close"], errors="coerce")
    aapl = aapl[["Close"]].dropna()
    goog = goog[["Close"]].dropna()
    goog = goog.rename(columns={"Close": "GOOG_Close"})

    data = aapl.merge(goog, left_index=True, right_index=True, how="inner")
    
    if data.empty:
        print("Error: No overlapping data.")
        return

    print(f"Merged data shape: {data.shape}")

    # Feature Engineering
    data["SMA20"] = data["Close"].rolling(20).mean()
    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
    weights = np.arange(1, 21)
    data["WMA20"] = data["Close"].rolling(20).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    data["STD20"] = data["Close"].rolling(20).std()
    data["Upper_Band"] = data["SMA20"] + 2 * data["STD20"]
    data["Lower_Band"] = data["SMA20"] - 2 * data["STD20"]

    returns = data[["Close", "GOOG_Close"]].pct_change()
    returns.columns = ["AAPL_Close", "GOOG_Close"]

    features_df = pd.concat(
        [returns, data[["SMA20", "EMA20", "WMA20", "Upper_Band", "Lower_Band"]]],
        axis=1
    ).dropna()
    
    print(f"Features shape: {features_df.shape}")

    # Strict Ordering
    expected_columns = list(scaler.feature_names_in_)
    print(f"Expected columns: {expected_columns}")
    
    features = features_df[expected_columns].astype(float)
    
    # Scaling
    scaled = scaler.transform(features)
    print("Scaling successful.")

    # Windowing
    time_step = 100
    if len(scaled) < time_step:
        print("Error: Not enough data for windowing.")
        return
        
    window = scaled[-time_step:].reshape(1, time_step, scaled.shape[1])
    print(f"Window shape: {window.shape}")

    # Prediction Loop
    print("Running prediction loop...")
    future_scaled = []
    for i in range(7):
        pred = model.predict(window, verbose=0)[0][0]
        future_scaled.append(pred)
        next_row = window[0, -1].copy()
        next_row[0] = pred
        window = np.concatenate([window[:, 1:, :], next_row.reshape(1, 1, -1)], axis=1)

    print(f"Raw predictions (scaled): {future_scaled}")

    # Inverse Transform
    dummy = np.zeros((7, scaled.shape[1]))
    dummy[:, 0] = future_scaled
    predicted_returns = scaler.inverse_transform(dummy)[:, 0]
    predicted_returns = np.clip(predicted_returns, -0.05, 0.05)
    
    print(f"Predicted Returns: {predicted_returns}")
    print("Verification PASS.")

if __name__ == "__main__":
    verify_inference()
