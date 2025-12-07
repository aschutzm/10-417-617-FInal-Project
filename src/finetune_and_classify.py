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

def finetune_and_classify(target_dataset="Cora"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    backbone_path = "checkpoints/backbone_arxiv_128.pt" 
    denoiser_path = "checkpoints/unified_denoiser_arxiv_128.pt"
    
    hidden_dim = 128
    lr = 0.001
    epochs = 200
    alpha_cls = 1.0   
    alpha_diff = 0.5  
    
    num_timesteps = 1000
    betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    print(f"Loading {target_dataset}...")
    data = load_regularized_dataset(target_dataset).to(device)
    num_classes = int(data.y.max().item()) + 1

    print(f"Loading Frozen GraphSAGE Encoder from {backbone_path}...")
    encoder = GNNEncoder(
        in_channels=128, 
        hidden_channels=128, 
        out_channels=128,
        model_type="graphsage" 
    ).to(device)
    
    if not os.path.exists(backbone_path):
        print(f"Error: Backbone checkpoint not found at {backbone_path}. Did you run Step 3?")
        return 0, 0, {}
    encoder.load_state_dict(torch.load(backbone_path, map_location=device))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    print(f"Loading Pre-trained Denoiser from {denoiser_path}...")
    denoiser = DiffusionDenoiser(hidden_dim=128).to(device)
    denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))
    
    classifier = nn.Linear(128, num_classes).to(device)
    
    optimizer = Adam(
        list(denoiser.parameters()) + list(classifier.parameters()),
        lr=lr
    )
    
    print(f"Starting Joint Fine-tuning on {target_dataset}...")
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    logs = {'loss': [], 'val_acc': [], 'test_acc': []}
    
    for epoch in range(1, epochs + 1):
        denoiser.train()
        classifier.train()
        optimizer.zero_grad()
        
        with torch.no_grad():
            z0 = encoder(data.x, data.edge_index)
            
        t = torch.randint(0, num_timesteps, (z0.size(0),), device=device).long()
        noise = torch.randn_like(z0)
        a_bar = alphas_cumprod[t].view(-1, 1)
        z_t = torch.sqrt(a_bar) * z0 + torch.sqrt(1 - a_bar) * noise
        
        noise_pred = denoiser(z_t, t)
        loss_diff = F.mse_loss(noise_pred, noise)
        
        z0_hat = (z_t - torch.sqrt(1 - a_bar) * noise_pred) / torch.sqrt(a_bar)
        logits = classifier(z0_hat)
        
        if hasattr(data, 'train_mask'):
            loss_cls = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        else:
            loss_cls = torch.tensor(0.0, device=device)
            
        loss = alpha_diff * loss_diff + alpha_cls * loss_cls
        
        loss.backward()
        optimizer.step()
        
        denoiser.eval()
        classifier.eval()
        with torch.no_grad():
            logits_val = classifier(z0)
            val_acc = accuracy(logits_val, data.y, data.val_mask)
            test_acc = accuracy(logits_val, data.y, data.test_mask)
            
            logs['loss'].append(loss.item())
            logs['val_acc'].append(val_acc)
            logs['test_acc'].append(test_acc)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                torch.save({
                    'denoiser': denoiser.state_dict(),
                    'classifier': classifier.state_dict()
                }, f"checkpoints/finetuned_model_{target_dataset.lower()}.pt")

    print(f"Finished. Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc, best_test_acc, logs

if __name__ == "__main__":
    finetune_and_classify()
