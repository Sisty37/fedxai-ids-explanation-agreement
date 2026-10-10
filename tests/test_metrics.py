import numpy as np, pytest
from scipy import stats
from fedxai.metrics.agreement import (importance, rbo_ext_batch, d_rbo, d_spearman, jaccard_chance,
    jaccard_topk, jaccard_star, bootstrap_importances, client_eed, eed_summary, factorized, t_ci)


def test_rbo_identity_and_symmetry():
    rng = np.random.default_rng(0)
    a, b = rng.random(70), rng.random(70)
    assert d_rbo(a, a) == pytest.approx(0, abs=1e-12)
    assert d_rbo(a, b) == pytest.approx(d_rbo(b, a))
    assert 0 <= d_rbo(a, b) <= 1


def test_rbo_is_top_weighted():
    s = np.arange(70, 0, -1).astype(float)          # feature 0 most important
    top = s.copy(); top[[0, 1]] = top[[1, 0]]        # swap ranks 1 and 2
    bot = s.copy(); bot[[68, 69]] = bot[[69, 68]]    # swap last two
    assert d_rbo(s, top) > 100 * d_rbo(s, bot)
    # Spearman-based divergence cannot tell these apart
    assert d_spearman(s, top) == pytest.approx(d_spearman(s, bot))


def test_rbo_batch_matches_bruteforce():
    rng = np.random.default_rng(1)
    a, b, p = rng.random(12), rng.random(12), 0.9
    oa, ob = np.argsort(-a), np.argsort(-b)
    F = 12
    X = [len(set(oa[:d]) & set(ob[:d])) for d in range(1, F + 1)]
    ref = (X[-1] / F) * p ** F + (1 - p) / p * sum(X[d - 1] / d * p ** d for d in range(1, F + 1))
    assert rbo_ext_batch(a, b, p)[0] == pytest.approx(ref)


@pytest.mark.parametrize("k,F,expected", [(3, 18, 0.1060), (5, 18, 0.1740), (10, 18, 0.3925), (15, 18, 0.7164)])
def test_jaccard_chance_matches_fedshap_wsn(k, F, expected):
    assert jaccard_chance(k, F) == pytest.approx(expected, abs=5e-5)


def test_jaccard_chance_monte_carlo():
    rng = np.random.default_rng(0)
    k, F = 10, 70
    sims = [jaccard_topk(rng.random(F), rng.random(F), k) for _ in range(20000)]
    assert np.mean(sims) == pytest.approx(jaccard_chance(k, F), abs=0.005)
    a = rng.random(F)
    assert jaccard_star(a, a, k) == pytest.approx(1.0)


def _phi(n, w, rng, noise=0.3):
    return np.abs(rng.normal(size=(n, len(w))) * noise + w)


def test_bootstrap_shapes():
    rng = np.random.default_rng(0)
    bt = bootstrap_importances(_phi(200, np.ones(20), rng), 250, rng)
    assert bt.shape == (250, 20) and np.allclose(bt.sum(1), 1)


def test_eed_null_vs_shift():
    rng = np.random.default_rng(0)
    w = np.linspace(3, 0.1, 40)
    ref = _phi(500, w, rng)
    same = [client_eed(_phi(500, w, rng), ref, B=300, n_pairs=2000, rng=rng) for _ in range(10)]
    w2 = w.copy(); w2[:8] = w2[:8][::-1]             # reorder the top features
    shifted = [client_eed(_phi(500, w2, rng), ref, B=300, n_pairs=2000, rng=rng) for _ in range(10)]
    s0, s1 = eed_summary(same), eed_summary(shifted)
    assert abs(s0["EED"]) < 0.05 and s0["sig_rate"] <= 0.2
    assert s1["EED"] > 0.1 and s1["sig_rate"] == 1.0
    assert s1["EED_plus"] >= s1["EED"]


def test_factorized_interaction():
    f = factorized(0.10, 0.05, 0.20)
    assert f["interaction"] == pytest.approx(0.05)


def test_t_ci():
    v = [0.1, 0.12, 0.09, 0.11, 0.13]
    m, h = t_ci(v)
    lo, hi = stats.t.interval(0.95, 4, loc=np.mean(v), scale=stats.sem(v))
    assert m == pytest.approx(np.mean(v)) and h == pytest.approx((hi - lo) / 2)
