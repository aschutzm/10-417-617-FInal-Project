This repository represents the implementation of the first stage of the Graph Diffusion Autoencoder (GDAE) project. The goal is to study how self-supervised pretraining on unlabeled graph data can help improve node classification performance. In this stage, a baseline Graph Neural Network classifier is implemented using PyTorch Geometric on benchmark citation datasets such as Cora, Citeseer, and PubMed.
Overview
The base GNN acts as a point of comparison for later experiments incorporating diffusion-based denoising models. It provides a basic approach to applying a GCN for node classification, reporting metrics such as accuracy and F1 score, and training loss per epoch.
Installation
Requirements:
Python 3.11+, PyTorch ≥ 2.0, PyTorch Geometric, NumPy, scikit-learn, tqdm
Install dependencies:
pip install torch torchvision torchaudio torch-geometric numpy scikit-learn tqdm
Usage
Train the model:
python train_gnn_classifier.py --dataset cora --epochs 200 --lr 0.01
Arguments:
--dataset: Name of dataset (cora, citeseer, pubmed)
--epochs: Number of epochs. Default is 200
--lr: Learning rate (default: 0.01)
--hidden_dim: Hidden dimension (by default: 64)
--dropout: Dropout rate (default: 0.5)
Example output:
Epoch 200 | Loss: 0.43 | Val Acc: 81.5% | Test F1: 0.79
Files
train_gnn_classifier.py: Training and evaluation logic
gnn_classifier.py: defines GCN model architecture
metrics.py: Helpers to calculate accuracy, F1 score, and loss
Project_Proposal.pdf: Research motivation and plan
Author
Ashton Schutzman
Carnegie Mellon University — B.S. Computer Science
