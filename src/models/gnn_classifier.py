from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm


class GNNClassifier(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        model: Literal["gcn", "graphsage"] = "gcn",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if model not in {"gcn", "graphsage"}:
            raise ValueError("model must be 'gcn' or 'graphsage'")

        Conv = GCNConv if model == "gcn" else SAGEConv

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(Conv(in_channels, hidden_channels))
        self.norms.append(BatchNorm(hidden_channels))

        for _ in range(max(0, num_layers - 2)):
            self.convs.append(Conv(hidden_channels, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        logits = self.head(x)
        return logits

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_gnn_classifier(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 128,
    num_layers: int = 2,
    dropout: float = 0.5,
    model: Literal["gcn", "graphsage"] = "gcn",
) -> GNNClassifier:
    return GNNClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
        model=model,
    )



