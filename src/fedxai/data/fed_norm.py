"""Federated computation of global normalisation statistics (not a privacy mechanism).

Round 1: each client sends per-column min -> server decides log1p columns
         (global min >= 0).
Round 2: each client sends (n, sum x, sum x^2) of the transformed data ->
         server returns global mean/std. Cross-dataset: reuse the 2017 stats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormStats:
    log1p_mask: np.ndarray  # (F,) bool
    mean: np.ndarray        # (F,)
    std: np.ndarray         # (F,)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).copy()
        m = self.log1p_mask
        # clip at 0 so out-of-range values in shifted data cannot produce NaN
        X[:, m] = np.log1p(np.clip(X[:, m], 0, None))
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"log1p_mask": self.log1p_mask.tolist(), "mean": self.mean.tolist(), "std": self.std.tolist()}


def client_min(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64).min(axis=0)


def client_moments(X: np.ndarray, log1p_mask: np.ndarray):
    X = np.asarray(X, dtype=np.float64).copy()
    X[:, log1p_mask] = np.log1p(np.clip(X[:, log1p_mask], 0, None))
    return len(X), X.sum(axis=0), (X ** 2).sum(axis=0)


def federated_norm_stats(client_X: list[np.ndarray], use_log1p: bool = True, eps: float = 1e-8) -> NormStats:
    gmin = np.min([client_min(X) for X in client_X], axis=0)
    mask = (gmin >= 0) if use_log1p else np.zeros_like(gmin, dtype=bool)
    n = s1 = s2 = 0
    for X in client_X:
        ni, a, b = client_moments(X, mask)
        n, s1, s2 = n + ni, s1 + a, s2 + b
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    std[std < eps] = 1.0
    return NormStats(mask, mean, std)


def build_reference(y_test_a: np.ndarray, per_class: int, background_size: int, seed: int):
    """R: up to `per_class` per class from Test-A. Background: stratified from R,
    equal per class where possible. Returns indices INTO Test-A."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y_test_a)
    R = np.concatenate([rng.permutation(np.flatnonzero(y_test_a == c))[:per_class] for c in classes])
    yR = y_test_a[R]
    q, r = divmod(background_size, len(classes))
    bg = []
    for i, c in enumerate(classes):
        pool = rng.permutation(R[yR == c])
        bg.append(pool[: q + (1 if i < r else 0)])
    bg = np.concatenate(bg)
    if len(bg) < background_size:  # tiny classes: top up from the rest of R
        rest = np.setdiff1d(R, bg)
        bg = np.concatenate([bg, rng.permutation(rest)[: background_size - len(bg)]])
    return np.sort(R), np.sort(bg)
