"""Load processed arrays, apply the federated normalisation stats, move to device."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..data.fed_norm import NormStats

CLASSES = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot"]


def load_2017(out_dir: str, device: str):
    z = np.load(os.path.join(out_dir, "cicids2017.npz"), allow_pickle=False)
    ns = json.load(open(os.path.join(out_dir, "norm_stats_full.json")))
    stats = NormStats(np.array(ns["log1p_mask"]), np.array(ns["mean"]), np.array(ns["std"]))
    X = torch.from_numpy(stats.transform(z["X"])).to(device)
    cls = {c: i for i, c in enumerate(CLASSES)}
    y = torch.from_numpy(np.array([cls[v] for v in z["y"]], dtype=np.int64)).to(device)
    idx = {k: torch.from_numpy(z[k].astype(np.int64)).to(device)
           for k in ["train", "test_a", "test_b", "R", "background"]}
    return X, y, idx, list(z["features"])


def load_partition(out_dir: str, name: str, device: str, n_clients: int = 10):
    z = np.load(os.path.join(out_dir, "partitions", name + ".npz"), allow_pickle=False)
    tr = [torch.from_numpy(z[f"train_{k}"].astype(np.int64)).to(device) for k in range(n_clients)]
    tb = [torch.from_numpy(z[f"testb_{k}"].astype(np.int64)).to(device) for k in range(n_clients)]
    return tr, tb


def class_weights(y_train: torch.Tensor, n_classes: int = len(CLASSES), cap: float = 50.0):
    """Square-root inverse-frequency weights (plain inverse would give ~13,000:1).
    In FL these come from summed client class counts, so they are identical."""
    n = torch.bincount(y_train, minlength=n_classes).double()
    w = torch.sqrt(n.sum() / (n_classes * n.clamp(min=1)))
    return w.clamp(max=cap).float()
