"""Explanation-agreement metrics (research plan, 'Explanation protocol and metrics').

Inputs are SHAP matrices phi (n_samples, n_features) for the explained class
logit(s). Everything works on |phi|.

Metric hierarchy: main = signed EED on RBO divergence; secondary = Spearman
divergence; validation = chance-corrected top-k Jaccard (J*).
"""
from __future__ import annotations

from math import comb

import numpy as np
from scipy import stats


# ---------- importance vectors ----------

def importance(phi: np.ndarray) -> np.ndarray:
    """Mean |phi| per feature, normalised to sum 1."""
    v = np.abs(np.asarray(phi, dtype=np.float64)).mean(axis=0)
    t = v.sum()
    return v / t if t > 0 else np.full_like(v, 1.0 / len(v))


def _positions(scores: np.ndarray) -> np.ndarray:
    """scores (P, F) -> pos (P, F): position of each feature in the descending
    ranking (0 = most important). Ties broken by feature index (stable)."""
    scores = np.atleast_2d(scores)
    order = np.argsort(-scores, axis=1, kind="stable")
    pos = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    pos[rows, order] = np.arange(order.shape[1])[None, :]
    return pos


# ---------- RBO ----------

def rbo_ext_batch(a: np.ndarray, b: np.ndarray, p: float = 0.9) -> np.ndarray:
    """Extrapolated Rank-Biased Overlap (Webber et al., 2010) for complete
    rankings of the same F features. a, b: score arrays (P, F) or (F,).
    With complete lists X_F = F, so RBO_ext = p^F + (1-p)/p * sum_d (X_d/d) p^d."""
    pa, pb = _positions(a), _positions(b)
    P, F = pa.shape
    depth = np.maximum(pa, pb)  # feature is in both top-(d+1) prefixes iff depth <= d
    counts = np.zeros((P, F))
    np.add.at(counts, (np.repeat(np.arange(P), F), depth.ravel()), 1.0)
    X = np.cumsum(counts, axis=1)                      # X_d, d = 1..F
    d = np.arange(1, F + 1)
    w = p ** d
    return (X[:, -1] / F) * p ** F + (1 - p) / p * ((X / d) * w).sum(axis=1)


def d_rbo(a, b, p: float = 0.9):
    r = 1.0 - rbo_ext_batch(a, b, p)
    return float(r[0]) if r.shape[0] == 1 else r


def d_spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho = stats.spearmanr(a, b).statistic
    return float((1.0 - rho) / 2.0)


# ---------- chance-corrected top-k Jaccard (validation) ----------

def jaccard_chance(k: int, F: int) -> float:
    """Exact E[J] of two independent uniform k-subsets of F items
    (hypergeometric; FedSHAP-WSN, Proposition 1)."""
    tot = comb(F, k)
    return sum(m / (2 * k - m) * comb(k, m) * comb(F - k, k - m) / tot for m in range(k + 1))


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int) -> float:
    A = set(np.argsort(-a, kind="stable")[:k])
    B = set(np.argsort(-b, kind="stable")[:k])
    return len(A & B) / len(A | B)


def jaccard_star(a: np.ndarray, b: np.ndarray, k: int) -> float:
    j0 = jaccard_chance(k, len(a))
    return (jaccard_topk(a, b, k) - j0) / (1 - j0)


# ---------- bootstrap noise floor and EED ----------

def bootstrap_importances(phi: np.ndarray, B: int, rng, chunk: int = 100) -> np.ndarray:
    """Resample explained instances with replacement B times; reuse the SHAP
    matrix (no SHAP recomputation). Returns (B, F) normalised importances."""
    A = np.abs(np.asarray(phi, dtype=np.float64))
    n = len(A)
    out = []
    for s in range(0, B, chunk):
        idx = rng.integers(0, n, size=(min(chunk, B - s), n))
        m = A[idx].mean(axis=1)
        out.append(m / m.sum(axis=1, keepdims=True))
    return np.vstack(out)


def within_divergences(boot: np.ndarray, n_pairs: int, rng, p: float = 0.9) -> np.ndarray:
    """D between random distinct bootstrap pairs (b != b')."""
    B = len(boot)
    i = rng.integers(0, B, n_pairs)
    j = (i + rng.integers(1, B, n_pairs)) % B
    return 1.0 - rbo_ext_batch(boot[i], boot[j], p)


def client_eed(phi_client: np.ndarray, phi_ref: np.ndarray, *, p=0.9, B=1000,
               n_pairs=5000, rng=None, ref_within: np.ndarray | None = None) -> dict:
    """Signed EED for one client against the reference.
    D_i = D(s_i, s_R); floor = (N_i + N_R)/2; EED_i = D_i - floor.
    `significant`: D_i above the 95th percentile of the pooled within-bootstrap
    D values of the client and the reference (approximate null)."""
    rng = rng or np.random.default_rng(0)
    s_i, s_R = importance(phi_client), importance(phi_ref)
    D = d_rbo(s_i, s_R, p)
    w_i = within_divergences(bootstrap_importances(phi_client, B, rng), n_pairs, rng, p)
    if ref_within is None:
        ref_within = within_divergences(bootstrap_importances(phi_ref, B, rng), n_pairs, rng, p)
    N_i, N_R = float(w_i.mean()), float(ref_within.mean())
    floor = 0.5 * (N_i + N_R)
    thr = float(np.quantile(np.concatenate([w_i, ref_within]), 0.95))
    return {"D": D, "D_spearman": d_spearman(s_i, s_R), "N_i": N_i, "N_R": N_R,
            "floor": floor, "EED": D - floor, "thr95": thr, "significant": bool(D > thr),
            "n_explained": int(len(phi_client))}


def eed_summary(per_client: list[dict]) -> dict:
    e = np.array([c["EED"] for c in per_client])
    return {"EED": float(e.mean()),                       # primary: signed
            "EED_plus": float(np.maximum(e, 0).mean()),   # secondary, descriptive
            "sig_rate": float(np.mean([c["significant"] for c in per_client])),
            "n_clients": len(per_client)}


# ---------- 2x2 factorized analysis (NOT causal) ----------

def factorized(D_data: float, D_model: float, D_combined: float) -> dict:
    """Data = D(global|client, global|R); Model = D(local|R, global|R);
    Combined = D(local|client, global|R); Interaction = Combined - Data - Model."""
    return {"data": D_data, "model": D_model, "combined": D_combined,
            "interaction": D_combined - D_data - D_model}


# ---------- seed-level statistics ----------

def t_ci(values, conf: float = 0.95) -> tuple[float, float]:
    """Mean and t-based CI half-width across seeds (df = n-1)."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n < 2:
        return float(v.mean()), float("nan")
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return float(v.mean()), float(h)
