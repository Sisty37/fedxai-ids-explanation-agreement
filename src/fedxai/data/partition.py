"""Client partitions. Train is split across clients; Test-B is split with the SAME
per-class client proportions, so every client has a held-out local test set
drawn from its own label distribution (explanations are computed there)."""
from __future__ import annotations

import numpy as np


def _split_by_props(idx: np.ndarray, props: np.ndarray, rng) -> list[np.ndarray]:
    idx = rng.permutation(idx)
    cuts = (np.cumsum(props)[:-1] * len(idx)).round().astype(int)
    return np.split(idx, cuts)


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, seed: int,
                        min_size: int = 500, max_tries: int = 1000):
    """Returns (client_indices, P) where P[k, c] = share of class c given to client k."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for _ in range(max_tries):
        P = rng.dirichlet(np.full(n_clients, alpha), size=len(classes)).T  # (K, C)
        parts = [[] for _ in range(n_clients)]
        for ci, c in enumerate(classes):
            for k, s in enumerate(_split_by_props(np.flatnonzero(y == c), P[:, ci], rng)):
                parts[k].append(s)
        parts = [np.sort(np.concatenate(p)) for p in parts]
        if min(len(p) for p in parts) >= min_size:
            return parts, P
    raise RuntimeError(f"No partition with min_size={min_size} after {max_tries} tries (alpha={alpha})")


def attack_skew_partition(y: np.ndarray, n_clients: int, benign_label, seed: int,
                          families_per_client: int = 2):
    """Every client gets an equal Benign share + `families_per_client` attack
    families, rotating; each family is split equally among its holders."""
    rng = np.random.default_rng(seed)
    classes = list(np.unique(y))
    attacks = [c for c in classes if c != benign_label]
    holders = {c: [] for c in attacks}
    for k in range(n_clients):
        for j in range(families_per_client):
            holders[attacks[(k + j) % len(attacks)]].append(k)
    P = np.zeros((n_clients, len(classes)))
    P[:, classes.index(benign_label)] = 1.0 / n_clients
    for c, ks in holders.items():
        if len(ks) < 2:
            raise ValueError(f"Family {c} held by {len(ks)} client(s); need >= 2")
        P[ks, classes.index(c)] = 1.0 / len(ks)
    return apply_props(y, P, classes, rng), P


def apply_props(y: np.ndarray, P: np.ndarray, classes, rng) -> list[np.ndarray]:
    parts = [[] for _ in range(P.shape[0])]
    for ci, c in enumerate(classes):
        for k, s in enumerate(_split_by_props(np.flatnonzero(y == c), P[:, ci], rng)):
            parts[k].append(s)
    return [np.sort(np.concatenate(p)) for p in parts]


def local_test_split(y_test: np.ndarray, P: np.ndarray, classes, seed: int) -> list[np.ndarray]:
    """Split Test-B among clients with the train partition's class proportions.
    `classes` must be the column order of P (np.unique of the TRAIN labels)."""
    return apply_props(y_test, P, list(classes), np.random.default_rng(seed + 10_000))
