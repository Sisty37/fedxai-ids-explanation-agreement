"""Shared helpers for the analysis scripts."""
import json, os
import numpy as np, torch
from .data import CLASSES
from .model import MLP
from ..data.fed_norm import NormStats


def load_model(path, d, dev):
    m = MLP(d, len(CLASSES)).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=True)); m.eval()
    return m


def load_2018(out_dir, dev):
    z = np.load(os.path.join(out_dir, "cicids2018.npz"), allow_pickle=False)
    ns = json.load(open(os.path.join(out_dir, "norm_stats_full.json")))
    st = NormStats(np.array(ns["log1p_mask"]), np.array(ns["mean"]), np.array(ns["std"]))
    ci = {c: i for i, c in enumerate(CLASSES)}
    X8 = torch.from_numpy(st.transform(z["X"])).to(dev)
    y8 = torch.from_numpy(np.array([ci[v] for v in z["y"]], dtype=np.int64)).to(dev)
    return X8, y8, list(z["features"])
