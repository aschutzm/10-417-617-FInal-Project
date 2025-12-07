import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import sys
import copy

sys.path.append(os.getcwd())
from src.models.gnn_encoder import GNNEncoder
from src.models.diffusion_denoiser import DiffusionDenoiser
from src.data_utils import load_regularized_dataset
from src.utils.metrics import accuracy

def train_joint_pretrain(dataset_name="Arxiv"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    latent_dim = 128
    hidden_dim = 128 
    lr = 0.001
    epochs = 200
    dropout = 0.5
    
    lambda_cls = 1.0
    lambda_diff = 1.0
    
    print(f"Loading {dataset_name} for Joint Training...")
    try:
        data = load_regularized_dataset(dataset_name).to(device)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    num_classes = int(data.y.max().item()) + 1 if data.y is not None else 0
    
    encoder = GNNEncoder(
        in_channels=128, 
        hidden_channels=hidden_dim, 
        out_channels=latent_dim, 
        num_layers=2, 
        dropout=dropout,
        model_type="graphsage" 
    ).to(device)
    
    denoiser = DiffusionDenoiser(hidden_dim=latent_dim).to(device)
    
    if num_classes > 0:
        classifier = nn.Linear(latent_dim, num_classes).to(device)
    else:
        classifier = None

    num_timesteps = 1000
    betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    params = list(encoder.parameters()) + list(denoiser.parameters())
    if classifier:
        params += list(classifier.parameters())
        
    optimizer = Adam(params, lr=lr, weight_decay=5e-4)
    
    print("Starting Joint End-to-End Pre-training (GraphSAGE)...")
    
    for epoch in range(1, epochs + 1):
        encoder.train()
        denoiser.train()
        if classifier: classifier.train()
        
        optimizer.zero_grad()
        
        z0 = encoder(data.x, data.edge_index)
        
        t = torch.randint(0, num_timesteps, (z0.size(0),), device=device).long()
        noise = torch.randn_like(z0)
        a_bar = alphas_cumprod[t].view(-1, 1)
        z_t = torch.sqrt(a_bar) * z0 + torch.sqrt(1 - a_bar) * noise
        
        noise_pred = denoiser(z_t, t)
        loss_diff = F.mse_loss(noise_pred, noise)
        
        loss_cls = torch.tensor(0.0, device=device)
        if classifier and hasattr(data, 'train_mask') and data.train_mask.sum() > 0:
            logits = classifier(z0)
            y_target = data.y.squeeze() if data.y.dim() > 1 else data.y
            loss_cls = F.cross_entropy(logits[data.train_mask], y_target[data.train_mask])
            
        loss = lambda_diff * loss_diff + lambda_cls * loss_cls
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            log_str = f"Epoch {epoch:03d} | Total: {loss.item():.4f} | Diff: {loss_diff.item():.4f} | Cls: {loss_cls.item():.4f}"
            print(log_str)
            
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(encoder.state_dict(), "checkpoints/backbone_arxiv_128.pt")
    torch.save(denoiser.state_dict(), "checkpoints/unified_denoiser_arxiv_128.pt")
    print("Saved Jointly Trained Models (GraphSAGE).")

if __name__ == "__main__":
    train_joint_pretrain()
