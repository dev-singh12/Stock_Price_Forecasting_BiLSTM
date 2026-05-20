"""
Bahdanau (Additive) Attention layer using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Computes a context vector as a weighted sum of the LSTM hidden states.
    Uses additive attention:
        score = v^T tanh(W h_t)
        weights = softmax(score)
        context = sum(weights * h_t)
    """
    def __init__(self, units: int):
        super().__init__()
        self.W = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, time_steps, hidden_dim)
        Returns:
            context shape: (batch_size, hidden_dim)
        """
        # score shape: (batch_size, time_steps, 1)
        score = self.V(torch.tanh(self.W(x)))
        
        # attention_weights shape: (batch_size, time_steps, 1)
        attention_weights = F.softmax(score, dim=1)
        
        # context shape: (batch_size, time_steps, hidden_dim)
        context = attention_weights * x
        
        # sum over time axis -> (batch_size, hidden_dim)
        return torch.sum(context, dim=1)

if __name__ == "__main__":
    # Smoke test
    layer = BahdanauAttention(128)
    dummy_input = torch.randn(4, 100, 128)
    out = layer(dummy_input)
    print("Input shape: ", dummy_input.shape)
    print("Output shape:", out.shape)
    assert out.shape == (4, 128)
    print("BahdanauAttention smoke test PASSED")
