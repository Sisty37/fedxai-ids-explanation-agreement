"""Centralised training and FedAvg / FedProx simulation (sequential clients, one GPU)."""
from __future__ import annotations

import time

import torch
from torch import nn

from .model import get_params, set_params


def _epoch(model, X, y, idx, opt, loss_fn, batch, gen, mu=0.0, anchor=None):
    model.train()
    perm = idx[torch.randperm(len(idx), generator=gen, device="cpu").to(idx.device)]
    for s in range(0, len(perm), batch):
        b = perm[s:s + batch]
        loss = loss_fn(model(X[b]), y[b])
        if mu > 0:
            loss = loss + 0.5 * mu * sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def train_centralized(model, X, y, idx, epochs, weights, lr=1e-3, batch=256, seed=0, log=print):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for e in range(epochs):
        t = time.time()
        _epoch(model, X, y, idx, opt, loss_fn, batch, gen)
        log(f"[central] epoch {e + 1}/{epochs}  {time.time() - t:.1f}s")
    return model


def train_federated(model, X, y, client_idx, rounds, weights, *, mu=0.0, local_epochs=1,
                    lr=1e-3, batch=256, seed=0, checkpoints=(), on_checkpoint=None, log=print):
    """FedAvg (mu = 0) or FedProx (mu > 0); full participation; sample-size weighted
    averaging; fresh Adam per client per round. on_checkpoint(round, model) is
    called after the rounds listed in `checkpoints`."""
    gen = torch.Generator().manual_seed(seed)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    sizes = torch.tensor([len(c) for c in client_idx], dtype=torch.float32)
    share = (sizes / sizes.sum()).tolist()
    for r in range(1, rounds + 1):
        t = time.time()
        g = get_params(model)
        agg = [torch.zeros_like(p) for p in g]
        for k, idx in enumerate(client_idx):
            set_params(model, g)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(local_epochs):
                _epoch(model, X, y, idx, opt, loss_fn, batch, gen, mu, g)
            for a, p in zip(agg, model.parameters()):
                a.add_(p.detach(), alpha=share[k])
        set_params(model, agg)
        log(f"[fed mu={mu}] round {r}/{rounds}  {time.time() - t:.1f}s")
        if r in checkpoints and on_checkpoint:
            on_checkpoint(r, model)
    return model


def train_local(model, X, y, idx, epochs, weights, lr=1e-3, batch=256, seed=0):
    """Local-only baseline (E4): one client's data, same total epoch budget."""
    return train_centralized(model, X, y, idx, epochs, weights, lr, batch, seed, log=lambda *_: None)
