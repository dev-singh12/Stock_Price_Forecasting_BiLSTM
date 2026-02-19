import joblib
import pandas as pd
import numpy as np
import sys
import os

print(f"Python Executable: {sys.executable}")
print("--- Inspecting Scaler ---")
try:
    scaler = joblib.load('scaler.save')
    print(f"Scaler type: {type(scaler)}")
    if hasattr(scaler, 'feature_names_in_'):
        print(f"Feature Names: {list(scaler.feature_names_in_)}")
    else:
        print("Scaler has no feature_names_in_ attribute.")
    
    if hasattr(scaler, 'n_features_in_'):
        print(f"Number of Features: {scaler.n_features_in_}")
except Exception as e:
    print(f"Error loading scaler: {e}")

print("\n--- Inspecting Model ---")
try:
    # Try importing tensorflow inside the try block
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    model = load_model('bilstm_stock_model.keras')
    print(f"Model Input Shape: {model.input_shape}")
    model.summary()
except ImportError:
    print("TensorFlow not installed or not found. Skipping model inspection.")
except Exception as e:
    print(f"Error loading model: {e}")
