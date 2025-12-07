import sys
import os

sys.path.append(os.getcwd())
from src.train_gnn_classifier import train, TrainConfig

if __name__ == "__main__":
    print("Training with Denoised Embeddings...")
    cfg = TrainConfig(
        dataset="Cora",
        root="./data",
        model="gcn",
        hidden_channels=128, 
        denoised_embeddings="checkpoints/denoised_unified_cora_128.pt",
        out_dir="./checkpoints_denoised"
    )
    try:
        train(cfg)
    except Exception as e:
        print(f"Failed: {e}")
