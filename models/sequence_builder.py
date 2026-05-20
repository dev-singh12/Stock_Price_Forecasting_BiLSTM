"""
Sliding-window sequence builder for time-series training.

All operations are strictly chronological — no shuffling, no leakage.
"""
import numpy as np


def build_sequences(
    scaled: np.ndarray,
    window: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences from a scaled feature matrix.

    For a matrix of shape (N, F):
      X[i] = scaled[i : i + window]       shape (window, F)
      y[i] = scaled[i + window, 0]        log_return column at step i+window

    Total sequences: N - window
    Split: split_idx = int(0.8 * (N - window))
    X_train = X[:split_idx],  y_train = y[:split_idx]
    X_val   = X[split_idx:],  y_val   = y[split_idx:]

    NEVER shuffle. Chronological order is preserved throughout.

    Args:
        scaled: Scaled feature matrix, shape (N, F). Column 0 is log_return.
        window: Sequence length. Default 100.

    Returns:
        (X_train, y_train, X_val, y_val) as numpy arrays.

    Raises:
        ValueError: If N <= window (not enough rows to form any sequence).
    """
    N = scaled.shape[0]
    if N <= window:
        raise ValueError(f"Not enough rows: need > {window}, got {N}")
    
    total_seqs = N - window
    X = np.empty((total_seqs, window, scaled.shape[1]))
    y = np.empty((total_seqs,))
    
    for i in range(total_seqs):
        X[i] = scaled[i : i + window]
        y[i] = scaled[i + window, 0]
        
    split_idx = int(0.8 * total_seqs)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    return X_train, y_train, X_val, y_val


def get_scaler_fit_boundary(n_rows: int, window: int = 100) -> int:
    """
    Return the row index (exclusive) up to which the scaler should be fit.

    The scaler must be fit only on rows that contribute to training sequences.
    Training sequence X[split_idx-1] uses rows (split_idx-1) through
    (split_idx + window - 2) inclusive. Therefore the fit boundary is
    split_idx + window.

    Formula:
        split_idx = int(0.8 * (n_rows - window))
        boundary  = split_idx + window

    The scaler is fit on raw_features[:boundary]. This ensures no
    validation-period statistics influence the scaler's median and IQR.

    Args:
        n_rows: Total number of rows in the feature DataFrame.
        window: Sequence length. Default 100.

    Returns:
        int: The exclusive upper row index for scaler fitting.

    Raises:
        ValueError: If n_rows <= window.
    """
    if n_rows <= window:
        raise ValueError(f"Not enough rows: need > {window}, got {n_rows}")
    
    split_idx = int(0.8 * (n_rows - window))
    boundary = split_idx + window
    return boundary


if __name__ == "__main__":
    m = np.random.randn(250, 10)
    Xtr, ytr, Xv, yv = build_sequences(m, window=100)
    b = get_scaler_fit_boundary(250, window=100)
    print("X_train shape:", Xtr.shape)
    print("y_train shape:", ytr.shape)
    print("X_val shape:", Xv.shape)
    print("y_val shape:", yv.shape)
    print("Boundary:", b)
