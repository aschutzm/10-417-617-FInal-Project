from torch import nn
from torch import Tensor
from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm

class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int, 
        num_layers: int = 2,
        dropout: float = 0.5,
        model_type: str = "gcn",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        Conv = GCNConv if model_type == "gcn" else SAGEConv
        
        self.convs.append(Conv(in_channels, hidden_channels))
        self.norms.append(BatchNorm(hidden_channels))
        
        for _ in range(max(0, num_layers - 2)):
            self.convs.append(Conv(hidden_channels, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))
            
        if num_layers > 1:
            self.convs.append(Conv(hidden_channels, out_channels))
            self.norms.append(BatchNorm(out_channels)) 
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.num_layers = num_layers

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < self.num_layers - 1: 
                x = self.activation(x)
                x = self.dropout(x)
        return x
