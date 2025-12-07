import torch
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import os
import sys
import builtins

def mock_input(prompt=None):
    return 'n'
builtins.input = mock_input

_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load

def load_regularized_dataset(name, root="./data"):
    reg_path = os.path.join(root, "regularized_datasets", f"{name}_regularized.pt")
    if not os.path.exists(reg_path):
        reg_path = os.path.join(root, "regularized_datasets", f"{name.capitalize()}_regularized.pt")
        
    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Regularized features not found for {name} at {reg_path}")
        
    reg_x = torch.load(reg_path)
    
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=name)
        data = dataset[0]
    elif name.lower() == 'arxiv' or name.lower() == 'ogbn-arxiv':
        try:
            try:
                dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
                data = dataset[0]
                split_idx = dataset.get_idx_split()
                train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
            except:
                print("OGB load failed, trying direct processed file...")
                proc_path = os.path.join(root, "ogbn_arxiv", "processed", "geometric_data_processed.pt")
                if os.path.exists(proc_path):
                    data_list = torch.load(proc_path)
                    data = data_list[0] if isinstance(data_list, (tuple, list)) else data_list
                    train_idx = None
                else:
                     raise FileNotFoundError("Could not find OGBN-Arxiv data.")

            num_nodes = data.num_nodes
            
            if 'train_idx' in locals() and train_idx is not None:
                 data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                 data.train_mask[train_idx] = True
                 data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                 if 'val_idx' in locals(): data.val_mask[val_idx] = True
                 data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                 if 'test_idx' in locals(): data.test_mask[test_idx] = True
            else:
                 print("Using RANDOM splits for Arxiv (Validation/Test ignored for pre-training).")
                 indices = torch.randperm(num_nodes)
                 n_train = int(0.8 * num_nodes) 
                 
                 data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                 data.train_mask[indices[:n_train]] = True
                 
                 data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                 data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        except Exception as e:
            print(f"Critical error loading Arxiv: {e}")
            raise e
            
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if data.num_nodes != reg_x.shape[0]:
        print(f"Warning: Node count mismatch for {name}. Graph: {data.num_nodes}, Reg: {reg_x.shape[0]}")
        
    data.x = reg_x
    return data

if __name__ == "__main__":
    for name in ["Cora", "Citeseer", "PubMed", "Arxiv"]:
        try:
            data = load_regularized_dataset(name)
            print(f"Loaded {name}: Nodes={data.num_nodes}, Features={data.x.shape}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
