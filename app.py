import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import logging
import os
import sys

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
MODEL_PATH = "bilstm_stock_model.keras"
SCALER_PATH = "scaler.save"
AAPL_CSV = "AAPL_data.csv"
GOOG_CSV = "GOOG_data.csv"

# Strict Schema Definition
EXPECTED_FEATURES = [
    'AAPL_Close', 'GOOG_Close', 'SMA20', 'EMA20', 'WMA20', 'Upper_Band', 'Lower_Band'
]
TIME_STEP = 100
FORECAST_DAYS = 7

# App Layout
st.set_page_config(
    page_title="ProStock AI Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Helper Functions (Robustness)
# -----------------------------------------------------------------------------

def validate_artifacts():
    """Checks existence of critical files at startup."""
    missing = []
    for path in [MODEL_PATH, SCALER_PATH, AAPL_CSV, GOOG_CSV]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        st.error(f"🚨 CRITICAL FAILURE: Missing artifacts: {missing}")
        logger.critical(f"Missing artifacts: {missing}")
        st.stop()
    logger.info("Artifact validation passed.")

def check_tensorflow():
    """Lazy import and version check."""
    try:
        import tensorflow as tf
        logger.info(f"Tensorflow Version: {tf.__version__}")
        return tf
    except ImportError:
        st.error("🚨 CRITICAL: TensorFlow not found.")
        st.stop()

# -----------------------------------------------------------------------------
# Resource Loading (Cached)
# -----------------------------------------------------------------------------

@st.cache_resource
def load_model_resources():
    """Loads Model and Scaler with error handling."""
    tf = check_tensorflow()
    from tensorflow.keras.models import load_model
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        
        logger.info(f"Loading scaler from {SCALER_PATH}...")
        scaler = joblib.load(SCALER_PATH)
        
        # Validate Scaler Schema
        if hasattr(scaler, 'feature_names_in_'):
            scaler_features = list(scaler.feature_names_in_)
            if scaler_features != EXPECTED_FEATURES:
                msg = f"Schema Mismatch! Scaler expects {scaler_features}, but app defines {EXPECTED_FEATURES}"
                logger.critical(msg)
                st.error(f"🚨 {msg}")
                st.stop()
        
        # Log Input Shape
        try:
            input_shape = model.input_shape
            logger.info(f"Model Input Shape: {input_shape}")
        except Exception:
            pass

        return model, scaler

    except Exception as e:
        logger.critical(f"Failed to load resources: {e}")
        st.error(f"Failed to load resources: {e}")
        st.stop()

@st.cache_data
def load_and_prep_data():
    """Loads CSVs, parses dates, ensures strictly numeric types, and aligns schema."""
    try:
        logger.info("Loading CSV data...")
        aapl = pd.read_csv(AAPL_CSV, index_col=0, parse_dates=True)
        goog = pd.read_csv(GOOG_CSV, index_col=0, parse_dates=True)

        # 1. Datetime Index Verification
        if not isinstance(aapl.index, pd.DatetimeIndex) or not isinstance(goog.index, pd.DatetimeIndex):
            raise ValueError("CSV index is not DatetimeIndex.")

        # 2. Numeric Enforcement
        aapl["Close"] = pd.to_numeric(aapl["Close"], errors="coerce")
        goog["Close"] = pd.to_numeric(goog["Close"], errors="coerce")
        aapl = aapl.dropna()
        goog = goog.dropna()

        # 3. Rename & Merge
        goog = goog.rename(columns={"Close": "GOOG_Close"})
        
        data = aapl[["Close"]].merge(
            goog[["GOOG_Close"]],
            left_index=True,
            right_index=True,
            how="inner"
        )

        if data.empty:
            raise ValueError("No overlapping dates found between AAPL and GOOG.")
        
        # 4. Feature Engineering (Strictly matching training)
        data["SMA20"] = data["Close"].rolling(20).mean()
        data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
        
        weights = np.arange(1, 21)
        data["WMA20"] = data["Close"].rolling(20).apply(
            lambda x: np.dot(x, weights) / weights.sum(),
            raw=True
        )

        data["STD20"] = data["Close"].rolling(20).std()
        data["Upper_Band"] = data["SMA20"] + 2 * data["STD20"]
        data["Lower_Band"] = data["SMA20"] - 2 * data["STD20"]

        # Returns
        returns = data[["Close", "GOOG_Close"]].pct_change()
        returns.columns = ["AAPL_Close", "GOOG_Close"]

        # 5. Final Feature Assembly
        features_df = pd.concat(
            [
                returns,
                data[["SMA20", "EMA20", "WMA20", "Upper_Band", "Lower_Band"]]
            ],
            axis=1
        ).dropna()
        
        # 6. Schema & Order Validation
        # STRICT ORDERING: ['AAPL_Close', 'GOOG_Close', 'SMA20', 'EMA20', 'WMA20', 'Upper_Band', 'Lower_Band']
        if list(features_df.columns) != EXPECTED_FEATURES:
            # Reorder if sets match
            if set(features_df.columns) == set(EXPECTED_FEATURES):
                features_df = features_df[EXPECTED_FEATURES]
            else:
                missing = set(EXPECTED_FEATURES) - set(features_df.columns)
                extra = set(features_df.columns) - set(EXPECTED_FEATURES)
                raise ValueError(f"Feature Schema Mismatch. Missing: {missing}, Extra: {extra}")

        # Ensure Float64
        features_df = features_df.astype('float64')

        # Keep track of recent price for un-scaling
        recent_price_data = data["Close"]

        logger.info(f"Data prepared successfully. Feature shape: {features_df.shape}")
        return features_df, recent_price_data

    except Exception as e:
        logger.error(f"Data Pipeline Failure: {e}")
        st.error(f"🚨 Data Pipeline Error: {e}")
        st.stop()


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------

def main():
    validate_artifacts()
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("🔧 Configuration")
        st.info(f"Model: BiLSTM v1.0\nTimestep: {TIME_STEP}\nForecast: {FORECAST_DAYS} days")
        
        show_debug = st.checkbox("Show Debug Info", value=False)
        st.markdown("---")
        st.markdown("### System Status")
        st.success("System Online")

    # --- Header ---
    st.title("📈 ProStock AI Forecast Engine")
    st.markdown("### AAPL Price Prediction System")
    st.markdown("Powered by **TensorFlow BiLSTM** & **Multi-Feature Analysis**")

    # --- Load Resources ---
    model, scaler = load_model_resources()
    features_df, recent_prices = load_and_prep_data()

    if len(features_df) < TIME_STEP:
        st.error(f"Insufficient data. Need {TIME_STEP} rows, got {len(features_df)}.")
        st.stop()

    # --- Run Inference ---
    if st.button("🚀 Generate 7-Day Forecast", type="primary"):
        with st.spinner("Running Inference Engine..."):
            try:
                # 1. Scale
                scaled_data = scaler.transform(features_df)
                
                # 2. Window
                window = scaled_data[-TIME_STEP:].reshape(1, TIME_STEP, len(EXPECTED_FEATURES))
                
                # 3. Forecast Loop
                future_scaled_preds = []
                current_window = window.copy()
                
                for _ in range(FORECAST_DAYS):
                    # PRED
                    pred = model.predict(current_window, verbose=0)[0][0]
                    future_scaled_preds.append(pred)
                    
                    # UPDATE WINDOW
                    next_row = current_window[0, -1].copy()
                    next_row[0] = pred # AAPL_Close return update
                    
                    # Reshape for concat
                    next_row_reshaped = next_row.reshape(1, 1, -1)
                    current_window = np.concatenate([current_window[:, 1:, :], next_row_reshaped], axis=1)
                
                # 4. Inverse Transform
                # We need to inverse transform the RETURNS.
                # Creates a dummy array to satisfy scaler shape
                dummy = np.zeros((FORECAST_DAYS, len(EXPECTED_FEATURES)))
                dummy[:, 0] = future_scaled_preds # Feature 0 is AAPL_Close
                
                # Inverse
                inv_trans = scaler.inverse_transform(dummy)
                predicted_returns = inv_trans[:, 0]
                
                # Clip returns for stability
                predicted_returns = np.clip(predicted_returns, -0.05, 0.05)
                
                # 5. Convert to Prices
                last_real_price = recent_prices.iloc[-1]
                forecast_prices = []
                curr_price = last_real_price
                
                for r in predicted_returns:
                    curr_price = curr_price * (1 + r)
                    forecast_prices.append(curr_price)
                
                # --- Metrics & Display ---
                
                # Sentiment Analysis
                start_p = last_real_price
                end_p = forecast_prices[-1]
                pct_change = ((end_p - start_p) / start_p) * 100
                
                sentiment = "NEUTRAL"
                color = "off"
                if pct_change > 1.0:
                    sentiment = "BULLISH 🐂"
                    color = "normal" 
                elif pct_change < -1.0:
                    sentiment = "BEARISH 🐻"
                    color = "inverse"
                
                # Heuristic Confidence (based on volatility of recent 20 days)
                recent_volatility = features_df["AAPL_Close"].iloc[-20:].std()
                # If volatility is high (>2%), confidence is low.
                confidence_score = max(0, min(100, int((1 - (recent_volatility * 10)) * 100)))
                if recent_volatility < 0.01: confidence_score = 90
                elif recent_volatility < 0.02: confidence_score = 75
                else: confidence_score = 50

                st.markdown("### Forecast Results")
                
                # Columns for Metrics
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Current Price", f"${last_real_price:.2f}")
                kpi2.metric("7-Day Target", f"${end_p:.2f}", f"{pct_change:.2f}%")
                kpi3.metric("Model Confidence", f"{confidence_score}%", sentiment)

                # --- Plotting ---
                dates_future = pd.date_range(start=recent_prices.index[-1], periods=FORECAST_DAYS + 1, freq="B")[1:]
                
                # Create DataFrame for Chart
                hist_data = recent_prices.iloc[-60:]
                
                forecast_df = pd.DataFrame({"Forecast": forecast_prices}, index=dates_future)
                
                # Merge for plotting
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(hist_data.index, hist_data.values, label="History", color="gray", alpha=0.7)
                ax.plot(dates_future, forecast_prices, label="Forecast", color="#00FF00" if pct_change > 0 else "#FF0000", marker="o", linestyle="--")
                
                ax.set_title(f"AAPL 7-Day Forecast ({sentiment})")
                ax.legend()
                ax.grid(True, alpha=0.2)
                st.pyplot(fig)
                
                # --- Download Artifacts ---
                csv_data = forecast_df.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Download Forecast CSV",
                    csv_data,
                    "forecast.csv",
                    "text/csv",
                    key='download-csv'
                )
                
                if show_debug:
                    st.warning("Debug Info Triggered")
                    st.json({
                        "Model Input Shape": str(model.input_shape),
                        "Scaler Features": list(scaler.feature_names_in_),
                        "Last Feature Row": features_df.iloc[-1].to_dict(),
                        "Predicted Returns": predicted_returns.tolist()
                    })
                
                logger.info("Inference completed successfully.")

            except Exception as e:
                logger.error(f"Inference Error: {e}")
                st.error("An error occurred during inference. Check logs.")
                if show_debug:
                    st.exception(e)

if __name__ == "__main__":
    main()
