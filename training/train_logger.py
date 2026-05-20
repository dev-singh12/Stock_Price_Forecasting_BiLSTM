"""
Structured logging for retraining events.

Appends one JSON record per retrain run to logs/retrain_log.jsonl.
Each record is a single-line JSON object (newline-delimited JSON format).
"""
import json
from pathlib import Path


LOG_PATH = Path("logs/retrain_log.jsonl")


def log_retrain_event(event: dict) -> None:
    """
    Append a JSON record to logs/retrain_log.jsonl.

    Required fields in event:
        run_date (str)          — ISO date, e.g. "2025-01-15"
        model_version (str)     — e.g. "model_v20250115"
        train_samples (int)     — number of training sequences
        val_samples (int)       — number of validation sequences
        best_val_loss (float)   — best val_loss from EarlyStopping
        epochs_trained (int)    — total epochs run
        early_stopped (bool)    — True if EarlyStopping triggered before max epochs
        duration_seconds (float)— wall-clock time
        status (str)            — "success" or "failed"
        train_end_date (str)    — ISO date of last training row
        val_start_date (str)    — ISO date of first validation row

    Optional field (on failure only):
        error_message (str)

    Creates logs/ directory if it does not exist.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
