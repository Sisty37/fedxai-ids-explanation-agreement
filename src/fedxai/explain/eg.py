"""Expected gradients (Erion et al., 2021): the estimator behind SHAP's
GradientExplainer, implemented directly in torch for speed and reproducibility.
phi(x) = E_{x'~background, a~U(0,1)} [ (x - x') * grad_x f_t(x' + a (x - x')) ]
Completeness holds in expectation: sum_j phi_j(x) ~ f_t(x) - E[f_t(x')]."""
from __future__ import annotations

import numpy as np
import torch


def expected_gradients(model, X, background, target, nsamples=200, seed=0, chunk=128):
    """X (n, F), background (m, F), target (n,) class index per row -> phi (n, F) numpy."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        t = target[s:s + chunk]
        b = len(x)
        j = torch.randint(0, len(background), (b * nsamples,), generator=g).to(X.device)
        a = torch.rand(b * nsamples, 1, generator=g).to(X.device)
        xr = x.repeat_interleave(nsamples, 0)
        ref = background[j]
        xi = (ref + a * (xr - ref)).requires_grad_(True)
        f = model(xi).gather(1, t.repeat_interleave(nsamples).unsqueeze(1)).sum()
        grad, = torch.autograd.grad(f, xi)
        phi = ((xr - ref) * grad).reshape(b, nsamples, -1).mean(1)
        out.append(phi.detach().cpu().numpy())
    return np.concatenate(out)
