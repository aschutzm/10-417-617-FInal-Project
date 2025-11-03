from __future__ import annotations

import torch
from torch import Tensor
from sklearn.metrics import f1_score


def accuracy(logits: Tensor, y: Tensor, mask: Tensor | None = None) -> float:
    if mask is not None:
        logits = logits[mask]
        y = y[mask]
    preds = logits.argmax(dim=-1)
    correct = (preds == y).sum().item()
    total = y.numel()
    return correct / max(total, 1)


def macro_f1(logits: Tensor, y: Tensor, mask: Tensor | None = None) -> float:
    if mask is not None:
        logits = logits[mask]
        y = y[mask]
    preds = logits.argmax(dim=-1).detach().cpu().numpy()
    y_true = y.detach().cpu().numpy()
    return float(f1_score(y_true, preds, average="macro"))



