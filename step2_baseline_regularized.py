import sys
import os

sys.path.append(os.getcwd())
from src.train_gnn_classifier import train, TrainConfig

def run_step2():
    print("==================================================")
    print("STEP 2: Baseline GNN on Regularized Data (Cora)")
    print("==================================================")
    
    cfg = TrainConfig(
        dataset="Cora",
        root="./data",
        model="gcn",
        hidden_channels=128,
        epochs=200,
        lr=0.01,
        dropout=0.5,
        use_regularized=True,
        out_dir="./checkpoints/step2_regularized"
    )
    
    train(cfg)

if __name__ == "__main__":
    run_step2()
