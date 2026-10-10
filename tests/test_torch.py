"""Run on the training machine (needs torch)."""
import numpy as np, pytest
torch = pytest.importorskip("torch")
from fedxai.train.model import MLP, get_params
from fedxai.train.fl import train_centralized, train_federated
from fedxai.train.data import class_weights
from fedxai.explain.eg import expected_gradients


def _toy(n=600, d=6, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    y = (X[:, 0] > 0).long() + (X[:, 1] > 1).long()
    return X, y


def test_mlp_shapes():
    assert MLP(80, 7)(torch.randn(5, 80)).shape == (5, 7)


def test_class_weights_sqrt_inverse():
    y = torch.tensor([0] * 900 + [1] * 90 + [2] * 10)
    w = class_weights(y, 3)
    assert w[2] > w[1] > w[0]
    assert float(w[2] / w[0]) == pytest.approx(np.sqrt(90), rel=1e-4)


def test_fedavg_single_client_equals_centralized():
    X, y = _toy()
    idx = torch.arange(len(X))
    w = torch.ones(3)
    torch.manual_seed(0); a = MLP(6, 3, p=0.0)
    torch.manual_seed(0); b = MLP(6, 3, p=0.0)
    train_centralized(a, X, y, idx, 1, w, seed=1, log=lambda *_: None)
    train_federated(b, X, y, [idx], 1, w, seed=1, log=lambda *_: None)
    for p, q in zip(get_params(a), get_params(b)):
        assert torch.allclose(p, q, atol=1e-6)


def test_fedprox_runs_and_moves_less():
    X, y = _toy()
    parts = [torch.arange(0, 300), torch.arange(300, 600)]
    w = torch.ones(3)
    torch.manual_seed(0); m0 = MLP(6, 3); init = get_params(m0)
    torch.manual_seed(0); m1 = MLP(6, 3)
    train_federated(m0, X, y, parts, 1, w, mu=0.0, seed=2, log=lambda *_: None)
    train_federated(m1, X, y, parts, 1, w, mu=10.0, seed=2, log=lambda *_: None)
    dist = lambda m: sum(((p - q) ** 2).sum() for p, q in zip(get_params(m), init))
    assert dist(m1) < dist(m0)


def test_expected_gradients_completeness():
    X, y = _toy()
    torch.manual_seed(0); m = MLP(6, 3, p=0.0)
    train_centralized(m, X, y, torch.arange(len(X)), 3, torch.ones(3), log=lambda *_: None)
    x, bg, t = X[:40], X[100:200], y[:40]
    phi = expected_gradients(m, x, bg, t, nsamples=2000, seed=0)
    with torch.no_grad():
        m.eval()
        gap = (m(x).gather(1, t.unsqueeze(1)).squeeze(1) - m(bg)[:, t].mean(0)).numpy()
    rel = np.abs(phi.sum(1) - gap).mean() / np.abs(gap).mean()
    assert rel < 0.1


def test_cnn_shapes_and_eg():
    from fedxai.train.model import CNN1D
    m = CNN1D(80, 7)
    assert m(torch.randn(4, 80)).shape == (4, 7)
    m.eval()
    phi = expected_gradients(m, torch.randn(3, 80), torch.randn(10, 80), torch.tensor([0, 1, 2]), nsamples=20)
    assert phi.shape == (3, 80)


def test_conv1dmm_matches_torch_conv1d():
    from fedxai.train.model import Conv1dMM
    torch.manual_seed(0)
    mm = Conv1dMM(3, 4, 5); ref = torch.nn.Conv1d(3, 4, 5, padding=2)
    with torch.no_grad():
        # Linear weight layout (cout, cin*k) with window flattened as (C, k)
        ref.weight.copy_(mm.lin.weight.view(4, 3, 5)); ref.bias.copy_(mm.lin.bias)
    x = torch.randn(2, 10, 3)
    assert torch.allclose(mm(x), ref(x.transpose(1, 2)).transpose(1, 2), atol=1e-5)
