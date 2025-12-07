import sys
import os

sys.path.append(os.getcwd())
from src.train_gnn_classifier import train, TrainConfig

def run_step1():
    print("STEP 1: Baseline GNN on Original Data (Cora)")

    
    cfg = TrainConfig(
        dataset="Cora",
        root="./data",
        model="gcn",
        hidden_channels=64,
        epochs=200,
        lr=0.01,
        dropout=0.5,
        use_regularized=False, 
        out_dir="./checkpoints/step1_baseline"
    )
    
    train(cfg)

if __name__ == "__main__":
    run_step1()
