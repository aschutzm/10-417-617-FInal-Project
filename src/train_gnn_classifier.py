from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

sys.path.append(os.getcwd())
from src.models.gnn_classifier import create_gnn_classifier
from src.utils.metrics import accuracy, macro_f1
from src.data_utils import load_regularized_dataset


@dataclass
class TrainConfig:
    dataset: str = "Cora"
    root: str = "./data"
    model: str = "gcn"
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    epochs: int = 200
    lr: float = 0.02
    weight_decay: float = 5e-4
    early_stopping: int = 100
    denoised_embeddings: str | None = None
    use_regularized: bool = False  
    out_dir: str = "./checkpoints"


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(cfg: TrainConfig):
    if cfg.use_regularized:
        data = load_regularized_dataset(cfg.dataset, root=cfg.root)
        try:
            dataset = Planetoid(root=cfg.root, name=cfg.dataset)
        except:
            dataset = None
        return dataset, data
    else:
        dataset = Planetoid(root=cfg.root, name=cfg.dataset)
        data = dataset[0]

    if cfg.denoised_embeddings:
        emb = torch.load(cfg.denoised_embeddings, map_location="cpu")
        if emb.ndim != 2 or emb.size(0) != data.num_nodes:
             print(f"Warning: Embedding shape {emb.shape} vs Node count {data.num_nodes}")
        data.x = emb

    return dataset, data


def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    dataset, data = load_dataset(cfg)
    in_channels = data.x.size(1)
    
    if dataset is not None:
        out_channels = dataset.num_classes
    else:
        out_channels = int(data.y.max().item()) + 1

    model = create_gnn_classifier(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        model=cfg.model,
    ).to(device)

    data = data.to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_state: dict | None = None
    epochs_no_improve = 0
    
    logs = {'loss': [], 'val_acc': [], 'test_acc': []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            train_acc = accuracy(logits, data.y, data.train_mask)
            val_acc = accuracy(logits, data.y, data.val_mask)
            test_acc = accuracy(logits, data.y, data.test_mask)
            
            logs['loss'].append(loss.item())
            logs['val_acc'].append(val_acc)
            logs['test_acc'].append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if cfg.early_stopping and epochs_no_improve >= cfg.early_stopping:
            break

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(cfg.out_dir, exist_ok=True)
    reg_suffix = "_reg" if cfg.use_regularized else ""
    ckpt_path = os.path.join(cfg.out_dir, f"gnn_{cfg.model.lower()}_{cfg.dataset.lower()}{reg_suffix}.pt")
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, ckpt_path)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_acc = accuracy(logits, data.y, data.test_mask)
        test_f1 = macro_f1(logits, data.y, data.test_mask)

    print(f"Best Val Acc: {best_val_acc:.4f} | Final Test Acc: {test_acc:.4f} | Test F1 (macro): {test_f1:.4f}")
    print(f"Saved checkpoint to: {ckpt_path}")
    
    return best_val_acc, test_acc, logs

if __name__ == "__main__":
    cfg = TrainConfig(
        dataset="Cora",
        root="./data",
        model="gcn",
        hidden_channels=64,
        num_layers=2,
        dropout=0.5,
        epochs=200,
        lr=0.02,
        weight_decay=5e-4,
        early_stopping=100,
        denoised_embeddings=None,
        use_regularized=False, 
        out_dir="./checkpoints",
    )
    train(cfg)
