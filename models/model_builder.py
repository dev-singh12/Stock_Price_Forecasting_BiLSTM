"""
BiLSTM model factory with optional Bahdanau attention.

Builds a fresh model each call — does not load or modify any existing
artifact, including the legacy bilstm_stock_model.keras.
"""
from tensorflow import keras
from models.attention import BahdanauAttention


def build_model(use_attention: bool, input_shape: tuple) -> keras.Model:
    """
    Build and return a compiled BiLSTM model.

    Args:
        use_attention: If True, inserts BahdanauAttention after the second BiLSTM.
        input_shape:   Tuple (window_size, n_features), e.g. (100, 10).

    Architecture with attention (use_attention=True):
        Input(100, 10)
        → Bidirectional(LSTM(64, return_sequences=True))   output: (batch, 100, 128)
        → Bidirectional(LSTM(64, return_sequences=True))   output: (batch, 100, 128)
        → BahdanauAttention(units=64)                      output: (batch, 128)
        → Dense(1)                                         output: (batch, 1)

    Architecture without attention (use_attention=False):
        Input(100, 10)
        → Bidirectional(LSTM(64, return_sequences=True))   output: (batch, 100, 128)
        → Bidirectional(LSTM(64, return_sequences=False))  output: (batch, 128)
        → Dense(1)                                         output: (batch, 1)

    The model predicts the next-day LOG RETURN (not raw price).
    Compilation: Adam(lr=1e-3), loss="mse", metrics=["mae"]

    Returns:
        keras.Model: Compiled model ready for training.
    """
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(inputs)
    
    if use_attention:
        x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)
        x = BahdanauAttention(units=64)(x)
    else:
        x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False))(x)
        
    outputs = keras.layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
    
    model.summary()
    return model


if __name__ == "__main__":
    m1 = build_model(use_attention=True, input_shape=(100, 10))
    m2 = build_model(use_attention=False, input_shape=(100, 10))
    print("Model with attention built successfully")
    print("Model without attention built successfully")
