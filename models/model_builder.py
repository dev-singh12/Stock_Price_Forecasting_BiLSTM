"""
Factory for building the PyTorch BiLSTM model.
"""
import torch
import torch.nn as nn
from models.attention import BahdanauAttention

class BiLSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        
        self.lstm1 = nn.LSTM(
            input_size=input_dim, 
            hidden_size=64, 
            batch_first=True, 
            bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            input_size=128, 
            hidden_size=64, 
            batch_first=True, 
            bidirectional=True
        )
        
        if self.use_attention:
            self.attention = BahdanauAttention(128)
            self.fc = nn.Linear(128, 1)
        else:
            self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, time_steps, input_dim)
        """
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        
        if self.use_attention:
            out = self.attention(out)
        else:
            # Take the last time step
            out = out[:, -1, :]
            
        return self.fc(out)

def build_model(use_attention: bool = True, input_shape: tuple = (100, 10)) -> nn.Module:
    """
    Build and return the PyTorch model.
    """
    time_steps, input_dim = input_shape
    return BiLSTMForecaster(input_dim=input_dim, use_attention=use_attention)

if __name__ == "__main__":
    m1 = build_model(use_attention=True)
    m2 = build_model(use_attention=False)
    dummy_input = torch.randn(4, 100, 10)
    print("With attention output shape:", m1(dummy_input).shape)
    print("Without attention output shape:", m2(dummy_input).shape)
    print("Model builders OK")
