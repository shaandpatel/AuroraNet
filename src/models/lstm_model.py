"""
Defines PyTorch model architectures for Kp forecasting.
Currently includes:

- KpLSTM : Multi-layer LSTM for multivariate time-series forecasting

These models are used by train.py, evaluate.py, and infer.py.
"""

import torch.nn as nn


class KpLSTM(nn.Module):
    """
    Multi-layer LSTM model for forecasting the Kp index from solar-wind features.

    Args:
        input_size (int): Number of input features per timestep.
        hidden_size (int): Dimensionality of the LSTM hidden state.
        num_layers (int): Number of stacked LSTM layers.
        output_size (int): Number of output values (e.g., horizon).
        dropout (float): Dropout applied between LSTM layers.

    Forward Input Shape:
        x: (batch_size, seq_length, input_size)

    Forward Output Shape:
        (batch_size, output_size)
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM output: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)

        # Use final timestep representation
        last_hidden = out[:, -1, :]   # (batch_size, hidden_size)

        # Map to forecast horizon
        out = self.fc(last_hidden)    # (batch_size, output_size)

        return out
