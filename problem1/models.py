import torch
from torch import nn


class RecurrentClassifier(nn.Module):
    """HAR/adding classifier using an allowed PyTorch recurrent baseline."""

    def __init__(
        self,
        cell: str,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        recurrent_types = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
        }
        if cell not in recurrent_types:
            raise ValueError(f"cell must be one of {tuple(recurrent_types)}")
        self.cell = cell
        self.recurrent = recurrent_types[cell](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        sequence_output, state = self.recurrent(x)
        if self.cell == "LSTM":
            state = state[0]
        last_hidden = state[-1]
        return self.classifier(last_hidden)


class RecurrentRegressor(nn.Module):
    """Adding-problem regressor using an allowed PyTorch recurrent baseline."""

    def __init__(self, cell: str, input_size: int = 2, hidden_size: int = 64):
        super().__init__()
        recurrent_types = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        if cell not in recurrent_types:
            raise ValueError(f"cell must be one of {tuple(recurrent_types)}")
        self.cell = cell
        self.recurrent = recurrent_types[cell](
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, state = self.recurrent(x)
        if self.cell == "LSTM":
            state = state[0]
        return self.regressor(state[-1]).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
