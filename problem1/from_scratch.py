"""Student implementation points for Problem 1, Task 2.

Do not copy in library recurrent modules here. Implement each cell from its
equations, then use the assignment's parameter-copying and comparison tests in
the notebook to check equivalence against PyTorch.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _initialize_parameters(parameters, hidden_size: int) -> None:
    bound = 1 / math.sqrt(hidden_size)
    with torch.no_grad():
        for parameter in parameters:
            nn.init.uniform_(parameter, -bound, bound)


class MyRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        _initialize_parameters(self.parameters(), hidden_size)

    def forward(self, x, h):
        # x: (batch, input_size); h: (batch, hidden_size)
        input_term = F.linear(x, self.weight_ih, self.bias_ih)
        hidden_term = F.linear(h, self.weight_hh, self.bias_hh)
        return torch.tanh(input_term + hidden_term)


class MyLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        _initialize_parameters(self.parameters(), hidden_size)

    def forward(self, x, state):
        # x: (batch, input_size); h and c: (batch, hidden_size)
        h, c = state
        gates = F.linear(x, self.weight_ih, self.bias_ih)
        gates = gates + F.linear(h, self.weight_hh, self.bias_hh)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        next_c = forget_gate * c + input_gate * cell_gate
        next_h = output_gate * torch.tanh(next_c)
        return next_h, next_c


class MyGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        _initialize_parameters(self.parameters(), hidden_size)

    def forward(self, x, h):
        # x: (batch, input_size); h: (batch, hidden_size)
        input_gates = F.linear(x, self.weight_ih, self.bias_ih)
        hidden_gates = F.linear(h, self.weight_hh, self.bias_hh)
        input_reset, input_update, input_new = input_gates.chunk(3, dim=1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=1)
        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        new_gate = torch.tanh(input_new + reset_gate * hidden_new)
        return (1 - update_gate) * new_gate + update_gate * h
