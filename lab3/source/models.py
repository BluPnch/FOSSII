"""Recurrent cells written explicitly; PyTorch supplies autograd and linear layers."""

import torch
from torch import nn


class RecurrentRegressor(nn.Module):
    def __init__(self, kind, input_size=5, hidden_size=32):
        super().__init__()
        if kind not in {"RNN", "GRU", "LSTM"}:
            raise ValueError(f"Unknown cell: {kind}")
        self.kind = kind
        self.hidden_size = hidden_size
        gates = {"RNN": 1, "GRU": 3, "LSTM": 4}[kind]
        self.input_projection = nn.Linear(input_size, gates * hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, gates * hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = x.new_zeros(x.shape[0], self.hidden_size)
        c = torch.zeros_like(h)
        # Project every input at once; the hidden state still evolves sequentially.
        for projected in self.input_projection(x).unbind(dim=1):
            recurrent = self.hidden_projection(h)
            if self.kind == "RNN":
                h = torch.tanh(projected + recurrent)
            elif self.kind == "GRU":
                xr, xz, xn = projected.chunk(3, dim=-1)
                hr, hz, hn = recurrent.chunk(3, dim=-1)
                reset = torch.sigmoid(xr + hr)
                update = torch.sigmoid(xz + hz)
                candidate = torch.tanh(xn + reset * hn)
                h = update * h + (1 - update) * candidate
            else:
                i, f, g, o = (projected + recurrent).chunk(4, dim=-1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                h = torch.sigmoid(o) * torch.tanh(c)
        return self.output(h).squeeze(-1)
