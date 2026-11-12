import torch
from torch import nn
from torch.nn import functional as F


def _run_recurrent_unrolled(recurrent: nn.Module, x, state=None):
    if recurrent.bidirectional:
        raise NotImplementedError(
            "DirectML recurrent execution currently supports unidirectional layers"
        )
    if not recurrent.batch_first:
        x = x.transpose(0, 1)

    batch_size, sequence_length, _ = x.shape
    hidden_size = recurrent.hidden_size
    layer_input = x
    hidden_states = []
    cell_states = []

    for layer in range(recurrent.num_layers):
        weight_ih = getattr(recurrent, f"weight_ih_l{layer}")
        weight_hh = getattr(recurrent, f"weight_hh_l{layer}")
        bias_ih = getattr(recurrent, f"bias_ih_l{layer}", None)
        bias_hh = getattr(recurrent, f"bias_hh_l{layer}", None)
        input_gates = F.linear(layer_input, weight_ih, bias_ih)

        if state is None:
            hidden = x.new_zeros(batch_size, hidden_size)
            cell = x.new_zeros(batch_size, hidden_size)
        elif recurrent.mode == "LSTM":
            hidden = state[0][layer]
            cell = state[1][layer]
        else:
            hidden = state[layer]

        outputs = []
        for time_index in range(sequence_length):
            input_term = input_gates[:, time_index]
            hidden_term = F.linear(hidden, weight_hh, bias_hh)

            if recurrent.mode.startswith("RNN"):
                hidden = input_term + hidden_term
                hidden = (
                    torch.tanh(hidden)
                    if recurrent.nonlinearity == "tanh"
                    else torch.relu(hidden)
                )
            elif recurrent.mode == "LSTM":
                input_gate, forget_gate, cell_gate, output_gate = (
                    input_term + hidden_term
                ).chunk(4, dim=1)
                cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(
                    input_gate
                ) * torch.tanh(cell_gate)
                hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
            else:
                input_reset, input_update, input_new = input_term.chunk(3, dim=1)
                hidden_reset, hidden_update, hidden_new = hidden_term.chunk(3, dim=1)
                reset_gate = torch.sigmoid(input_reset + hidden_reset)
                update_gate = torch.sigmoid(input_update + hidden_update)
                new_gate = torch.tanh(input_new + reset_gate * hidden_new)
                hidden = (1 - update_gate) * new_gate + update_gate * hidden
            outputs.append(hidden)

        layer_input = torch.stack(outputs, dim=1)
        if layer < recurrent.num_layers - 1 and recurrent.dropout:
            layer_input = F.dropout(
                layer_input, p=recurrent.dropout, training=recurrent.training
            )
        hidden_states.append(hidden)
        if recurrent.mode == "LSTM":
            cell_states.append(cell)

    if recurrent.mode == "LSTM":
        final_state = (
            torch.stack(hidden_states),
            torch.stack(cell_states),
        )
    else:
        final_state = torch.stack(hidden_states)
    if not recurrent.batch_first:
        layer_input = layer_input.transpose(0, 1)
    return layer_input, final_state


def run_recurrent_module(recurrent: nn.Module, x, state=None):
    """Run native recurrent layers, unrolling them where DirectML lacks fused ops."""
    if x.device.type == "privateuseone":
        return _run_recurrent_unrolled(recurrent, x, state)
    return recurrent(x, state) if state is not None else recurrent(x)


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
        sequence_output, state = run_recurrent_module(self.recurrent, x)
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
        _, state = run_recurrent_module(self.recurrent, x)
        if self.cell == "LSTM":
            state = state[0]
        return self.regressor(state[-1]).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
