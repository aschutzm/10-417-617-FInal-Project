import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())
from src.models.diffusion_denoiser import DiffusionDenoiser

def train_diffusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embedding_path = "checkpoints/unified_encoder_gcn_arxiv_128.pt"
    hidden_dim = 128
    lr = 1e-3
    epochs = 100
    batch_size = 1024
    
    if not os.path.exists(embedding_path):
        print(f"Embeddings not found at {embedding_path}")
        return

    print(f"Loading embeddings from {embedding_path}")
    embeddings = torch.load(embedding_path).to(device)
    
    dataset = TensorDataset(embeddings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DiffusionDenoiser(hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    num_timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    print("Starting Diffusion Training on Arxiv embeddings...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for (x0,) in loader:
            batch_size = x0.size(0)
            
            t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
            
            epsilon = torch.randn_like(x0)
            
            a_bar = alphas_cumprod[t]
            a_bar = a_bar.view(-1, 1)
            
            x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * epsilon
            
            epsilon_pred = model(x_t, t)
            
            loss = nn.functional.mse_loss(epsilon_pred, epsilon)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")
            
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/unified_denoiser_arxiv_128.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved denoiser to: {save_path}")

if __name__ == "__main__":
    train_diffusion()
