"""
Bahdanau-style additive attention layer for the BiLSTM stock forecasting model.

This is a lightweight, stateless implementation. It learns which timesteps
within the 100-day input window matter most for the next-day return prediction.

Serializable via get_config() for .keras format compatibility.
"""
import tensorflow as tf
from tensorflow import keras


class BahdanauAttention(keras.layers.Layer):
    """
    Bahdanau additive attention.

    Input:  (batch, timesteps, features) — output of BiLSTM with return_sequences=True
    Output: (batch, features)            — weighted sum of hidden states

    Computation:
        score   = tanh(inputs @ W_h)              shape: (batch, timesteps, units)
        alpha   = softmax(score @ v, axis=1)      shape: (batch, timesteps, 1)
        context = sum(alpha * inputs, axis=1)     shape: (batch, features)

    Trainable weights:
        W_h: (features, units)
        v:   (units, 1)
    """

    def __init__(self, units: int = 64, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape: tuple) -> None:
        features = input_shape[-1]
        self.W_h = self.add_weight(
            name="W_h",
            shape=(features, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.v = self.add_weight(
            name="v",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs shape: (batch, timesteps, features)
        # inputs @ W_h -> (batch, timesteps, units)
        score = tf.nn.tanh(tf.matmul(inputs, self.W_h))
        # score @ v -> (batch, timesteps, 1)
        alpha = tf.nn.softmax(tf.matmul(score, self.v), axis=1)
        # alpha * inputs -> (batch, timesteps, features)
        # sum over axis=1 -> (batch, features)
        context = tf.reduce_sum(alpha * inputs, axis=1)
        return context

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"units": self.units})
        return config


if __name__ == "__main__":
    import numpy as np
    layer = BahdanauAttention(units=64)
    x = tf.random.normal((4, 100, 128))  # batch=4, 100 timesteps, 128 features
    out = layer(x)
    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)   # must be (4, 128)
    assert out.shape == (4, 128), f"Shape mismatch: {out.shape}"
    print("BahdanauAttention smoke test PASSED")
