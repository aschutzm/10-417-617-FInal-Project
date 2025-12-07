import torch
import os
import sys
import math

sys.path.append(os.getcwd())
from src.models.diffusion_denoiser import DiffusionDenoiser

def denoise_datasets():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    denoiser_path = "checkpoints/unified_denoiser_arxiv_128.pt"
    datasets = ["cora", "citeseer", "pubmed"]
    
    hidden_dim = 128
    num_timesteps = 1000
    start_t = 50 
    
    print(f"Loading denoiser from {denoiser_path}")
    model = DiffusionDenoiser(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(denoiser_path, map_location=device))
    model.eval()
    
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    for name in datasets:
        emb_path = f"checkpoints/unified_encoder_gcn_{name}_128.pt"
        if not os.path.exists(emb_path):
            print(f"Embedding for {name} not found at {emb_path}")
            continue
            
        print(f"Denoising {name}...")
        z0 = torch.load(emb_path).to(device)
        
        batch_size = z0.size(0)
        t_tensor = torch.full((batch_size,), start_t, device=device, dtype=torch.long)
        epsilon = torch.randn_like(z0)
        
        a_bar_t = alphas_cumprod[start_t]
        z_t = torch.sqrt(a_bar_t) * z0 + torch.sqrt(1 - a_bar_t) * epsilon
        
        x = z_t
        with torch.no_grad():
            for t in reversed(range(0, start_t + 1)):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                pred_noise = model(x, t_batch)
                
                alpha_t = alphas[t]
                beta_t = betas[t]
                alpha_bar_t = alphas_cumprod[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                
                coeff1 = 1 / torch.sqrt(alpha_t)
                coeff2 = beta_t / torch.sqrt(1 - alpha_bar_t)
                
                mean = coeff1 * (x - coeff2 * pred_noise)
                
                sigma = torch.sqrt(beta_t)
                
                x = mean + sigma * noise
                
        out_path = f"checkpoints/denoised_unified_{name}_128.pt"
        torch.save(x.cpu(), out_path)
        print(f"Saved denoised embeddings to {out_path}")

if __name__ == "__main__":
    denoise_datasets()
