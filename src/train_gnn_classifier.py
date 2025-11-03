from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

from models.gnn_classifier import create_gnn_classifier
from utils.metrics import accuracy, macro_f1


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
    out_dir: str = "./checkpoints"


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(cfg: TrainConfig):
    dataset = Planetoid(root=cfg.root, name=cfg.dataset)
    data = dataset[0]

    if cfg.denoised_embeddings:
        emb = torch.load(cfg.denoised_embeddings, map_location="cpu")
        if emb.ndim != 2 or emb.size(0) != data.num_nodes:
            raise ValueError(
                f"Loaded embeddings shape {tuple(emb.shape)} incompatible with num_nodes={data.num_nodes}"
            )
        data.x = emb

    return dataset, data


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    dataset, data = load_dataset(cfg)
    in_channels = data.x.size(1)
    out_channels = dataset.num_classes

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
    ckpt_path = os.path.join(cfg.out_dir, f"gnn_{cfg.model.lower()}_{cfg.dataset.lower()}.pt")
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, ckpt_path)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_acc = accuracy(logits, data.y, data.test_mask)
        test_f1 = macro_f1(logits, data.y, data.test_mask)

    print(f"Best Val Acc: {best_val_acc:.4f} | Final Test Acc: {test_acc:.4f} | Test F1 (macro): {test_f1:.4f}")
    print(f"Saved checkpoint to: {ckpt_path}")

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
        out_dir="./checkpoints",
    )
    train(cfg)



