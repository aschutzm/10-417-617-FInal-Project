import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import copy
from torch_geometric.loader import DataLoader
import sys
sys.path.append(os.getcwd())
from src.models.gnn_encoder import GNNEncoder
from src.data_utils import load_regularized_dataset
from src.utils.metrics import accuracy

def train_unified_gnn(datasets_list=['Arxiv'], out_backbone_name="backbone_arxiv_128.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    latent_dim = 128
    hidden_dim = 128 
    lr = 0.01
    epochs = 200
    dropout = 0.5
    
    datasets = {}
    for name in datasets_list:
        print(f"Loading {name}...")
        try:
            data = load_regularized_dataset(name)
            data = data.to(device)
            datasets[name] = data
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            return

    backbone = GNNEncoder(
        in_channels=128, 
        hidden_channels=hidden_dim, 
        out_channels=latent_dim, 
        num_layers=2, 
        dropout=dropout
    ).to(device)
    
    heads = nn.ModuleDict()
    for name, data in datasets.items():
        if data.y is not None:
            if data.y.dim() > 1 and data.y.size(1) == 1:
                 num_classes = int(data.y.max().item()) + 1
            else:
                 num_classes = int(data.y.max().item()) + 1
        else:
            num_classes = 0
            
        if num_classes > 0:
            heads[name] = nn.Linear(latent_dim, num_classes).to(device)
        else:
            print(f"Skipping head for {name} (no labels found)")

    optimizer = Adam(
        list(backbone.parameters()) + list(heads.parameters()), 
        lr=lr, 
        weight_decay=5e-4
    )
    
    best_val_acc = {name: 0.0 for name in heads.keys()}
    
    print(f"Starting Pre-training on {datasets_list}...")
    
    for epoch in range(1, epochs + 1):
        backbone.train()
        heads.train()
        optimizer.zero_grad()
        
        total_loss = 0
        dataset_stats = {}
        
        for name, data in datasets.items():
            if name not in heads:
                continue
                
            embeddings = backbone(data.x, data.edge_index)
            logits = heads[name](embeddings)
            
            mask = None
            if hasattr(data, 'train_mask'):
                mask = data.train_mask
            
            if mask is not None:
                y = data.y.squeeze()
                loss = F.cross_entropy(logits[mask], y[mask])
                total_loss += loss
                
                acc = accuracy(logits, y, mask)
                dataset_stats[name] = acc
            
        if total_loss == 0:
            if epoch == 1: 
                print("Warning: No supervision (gradients) for GNN training. Check masks.")
            break
            
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            log_str = f"Epoch {epoch:03d} | Loss {total_loss.item():.4f} | "
            for name in dataset_stats:
                data = datasets[name]
                if hasattr(data, 'val_mask'):
                    with torch.no_grad():
                        backbone.eval()
                        heads[name].eval()
                        emb = backbone(data.x, data.edge_index)
                        l = heads[name](emb)
                        y = data.y.squeeze()
                        val_acc = accuracy(l, y, data.val_mask)
                        test_acc = accuracy(l, y, data.test_mask)
                        log_str += f"{name}: Train {dataset_stats[name]:.3f} | Val {val_acc:.3f} | Test {test_acc:.3f} | "
                        
                        if val_acc > best_val_acc[name]:
                            best_val_acc[name] = val_acc
            print(log_str)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(backbone.state_dict(), f"checkpoints/{out_backbone_name}")
    print(f"Saved backbone to: checkpoints/{out_backbone_name}")
    
    backbone.eval()
    with torch.no_grad():
        for name, data in datasets.items():
            emb = backbone(data.x, data.edge_index)
            torch.save(emb.cpu(), f"checkpoints/unified_encoder_gcn_{name.lower()}_128.pt")
            print(f"Saved embeddings for {name}")

if __name__ == "__main__":
    train_unified_gnn(datasets_list=['Arxiv'], out_backbone_name="backbone_arxiv_128.pt")
