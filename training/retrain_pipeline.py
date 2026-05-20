"""
End-to-end AAPL model retraining pipeline.

Triggered manually via:
    python -m training.retrain_pipeline         (respects 7-day cooldown)
    python -m training.retrain_pipeline --force (always retrains)
"""
import time
import json
from datetime import datetime, date
from pathlib import Path
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from features.feature_builder import build_features
from models.sequence_builder import build_sequences, get_scaler_fit_boundary
from models.scaler_manager import ScalerManager
from models.model_builder import build_model
from training.train_logger import log_retrain_event

MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def run_retrain(force: bool = False) -> dict:
    today_str = date.today().isoformat()
    
    if not force:
        log_file = Path("logs/retrain_log.jsonl")
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                if lines:
                    try:
                        last_log = json.loads(lines[-1])
                        if last_log.get("run_date") == today_str and last_log.get("status") == "success":
                            return {"status": "skipped", "reason": "already retrained today"}
                    except json.JSONDecodeError:
                        pass
    
    start_time = time.time()
    version_date = date.today().strftime("%Y%m%d")
    model_version = f"model_v{version_date}"
    
    try:
        # Step 1: Read data/aapl_rolling.csv and build features
        rolling_df = pd.read_csv("data/aapl_rolling.csv", index_col=0, parse_dates=True)
        features_df = build_features(rolling_df)
        
        # Step 2 & 3: Compute scaler fit boundary
        n_rows = len(features_df)
        boundary = get_scaler_fit_boundary(n_rows, window=100)
        
        # Step 4: Fit and save scaler
        sm = ScalerManager()
        scaler = sm.fit_and_save(features_df, boundary, version_date)
        meta_path = MODELS_DIR / f"scaler_v{version_date}_meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # Step 5: Transform FULL feature matrix
        scaled_features = sm.transform(features_df, scaler, meta)
        
        # Step 6: Build sequences
        X_train, y_train, X_val, y_val = build_sequences(scaled_features, window=100)
        
        # Step 7: Torch setup
        device = get_device()
        print(f"Training on device: {device}")
        
        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_v = torch.tensor(X_val, dtype=torch.float32)
        y_v = torch.tensor(y_val, dtype=torch.float32)
        
        train_ds = TensorDataset(X_tr, y_tr)
        val_ds = TensorDataset(X_v, y_v)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        model = build_model(use_attention=True, input_shape=(100, 10)).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Step 8: Training Loop
        epochs = 200
        patience = 10
        best_val_loss = float('inf')
        patience_counter = 0
        epochs_trained = 0
        best_weights = None
        
        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx).squeeze(-1)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx).squeeze(-1)
                    val_loss += criterion(out, by).item() * bx.size(0)
            val_loss /= len(val_ds)
            
            epochs_trained += 1
            print(f"Epoch {epoch+1}/{epochs} - val_loss: {val_loss:.5f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
        early_stopped = (epochs_trained < epochs)
        
        # Step 9: Save model
        if best_weights is not None:
            model.load_state_dict(best_weights)
        
        # Save as .pt
        torch.save(model.state_dict(), MODELS_DIR / f"{model_version}.pt")
        
        # Step 10: Cleanup old artifacts
        _cleanup_old_artifacts()
        
        # Step 11: Log event
        event = {
            "run_date": today_str,
            "model_version": model_version,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "best_val_loss": float(best_val_loss),
            "epochs_trained": epochs_trained,
            "early_stopped": early_stopped,
            "duration_seconds": float(time.time() - start_time),
            "status": "success",
            "train_end_date": str(features_df.index[boundary-1].date()),
            "val_start_date": str(features_df.index[boundary].date())
        }
        log_retrain_event(event)
        return event

    except Exception as e:
        event = {
            "run_date": today_str,
            "model_version": model_version,
            "train_samples": 0,
            "val_samples": 0,
            "best_val_loss": 0.0,
            "epochs_trained": 0,
            "early_stopped": False,
            "duration_seconds": float(time.time() - start_time),
            "status": "failed",
            "train_end_date": "",
            "val_start_date": "",
            "error_message": str(e)
        }
        log_retrain_event(event)
        raise RuntimeError(str(e)) from e


def _cleanup_old_artifacts(models_dir: Path = MODELS_DIR, keep: int = 3) -> None:
    # PyTorch Models
    pt_files = sorted(models_dir.glob("model_v*.pt"))
    for f in pt_files[:-keep]:
        f.unlink(missing_ok=True)
        
    # Delete any lingering keras files from the previous attempts
    keras_files = sorted(models_dir.glob("model_v*.keras"))
    for f in keras_files:
        f.unlink(missing_ok=True)

    # Scalers + metadata
    scaler_files = sorted(models_dir.glob("scaler_v*.pkl"))
    for f in scaler_files[:-keep]:
        f.unlink(missing_ok=True)
        meta = models_dir / f"{f.stem}_meta.json"
        meta.unlink(missing_ok=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if a model was already trained today")
    args = parser.parse_args()
    result = run_retrain(force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
