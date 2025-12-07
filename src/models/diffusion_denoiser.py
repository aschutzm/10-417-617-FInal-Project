import torch
from torch import nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionDenoiser(nn.Module):
    def __init__(self, hidden_dim=128, time_dim=32, num_layers=3):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        self.layers = nn.ModuleList(layers)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = x + t_emb 
            x = layer(x) + x 
            
        return self.output_proj(x)
