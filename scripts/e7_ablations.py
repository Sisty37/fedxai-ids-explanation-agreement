"""E7 robustness checks (no new training).
1. Metric check: chance-corrected top-k Jaccard (k = 5, 10, 15) for the fidelity and
   heterogeneity comparisons, and its agreement with the RBO divergence.
2. SHAP background: explanations of the same client samples with a client-local
   background (100 rows of the client's own training data) vs the shared background;
   and per-class client-vs-R EED when clients use local backgrounds.
3. Explainer check: KernelSHAP vs expected gradients on 60 R samples (central_s0),
   if the `shap` package is installed (pip install shap).
Usage: python scripts/e7_ablations.py     (~10-20 min)
Output: results/analysis/e7_ablations.json + e7_ablations_summary.txt
"""
import itertools, json, os, sys, time
import numpy as np, torch, yaml
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, load_partition, CLASSES
from fedxai.train.common import load_model
from fedxai.explain.eg import expected_gradients
from fedxai.metrics.agreement import (importance, d_rbo, jaccard_star, bootstrap_importances,
                                      within_divergences, client_eed, t_ci)

NS, PER = 1000, 200
C = len(CLASSES)


def main():
    cfg = yaml.safe_load(open("configs/protocol.yaml")); ex = cfg["explanation"]; od = cfg["data"]["out_dir"]
    P, B, NP_, MINC = ex["rbo_p"], ex["bootstrap_B"], ex["noise_pairs"], ex["min_class_samples"]
    seeds, K = cfg["seeds"], cfg["partition"]["n_clients"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, idx, feats = load_2017(od, dev)
    R, bg = idx["R"], idx["background"]; yR = y[R].cpu().numpy(); Xbg = X[bg]
    cls = [c for c in range(C) if (yR == c).sum() >= MINC]; CN = [CLASSES[c] for c in cls]
    rng = np.random.default_rng(0); t0 = time.time(); res = {}
    log = lambda m: print(f"[{(time.time() - t0) / 60:4.1f} min] {m}", flush=True)
    run = lambda n: os.path.join("results", "runs", n, "final.pt")
    phi = {}
    for s in seeds:
        phi[f"cen{s}"] = expected_gradients(load_model(run(f"central_s{s}"), X.shape[1], dev), X[R], Xbg, y[R], nsamples=NS, seed=0)
        for a in ["100", "1", "0.1"]:
            phi[f"{a}_{s}"] = expected_gradients(load_model(run(f"fedavg_dir_a{a}_s{s}"), X.shape[1], dev), X[R], Xbg, y[R], nsamples=NS, seed=0)
    log("explanations on R done")

    # ---- 1. metric check ----
    imp = lambda k, c: importance(phi[k][yR == c])
    M = {"fidelity_1minusJstar": {}, "heterogeneity_1minusJstar": {}}
    rbo_vals, j_vals = [], []
    for kk in [5, 10, 15]:
        M["fidelity_1minusJstar"][kk], M["heterogeneity_1minusJstar"][kk] = {}, {}
        for a in ["100", "1", "0.1"]:
            M["fidelity_1minusJstar"][kk][a] = {cn: t_ci([np.mean([1 - jaccard_star(imp(f"{a}_{s}", c), imp(f"cen{s2}", c), kk)
                                                                   for s2 in seeds if s2 != s]) for s in seeds]) for c, cn in zip(cls, CN)}
            if a != "100":
                M["heterogeneity_1minusJstar"][kk][a] = {cn: t_ci([np.mean([1 - jaccard_star(imp(f"{a}_{s}", c), imp(f"100_{s2}", c), kk)
                                                                         for s2 in seeds if s2 != s]) for s in seeds]) for c, cn in zip(cls, CN)}
        M["fidelity_1minusJstar"][kk]["central_baseline"] = {cn: float(np.mean([1 - jaccard_star(imp(f"cen{i}", c), imp(f"cen{j}", c), kk)
                                                                                 for i, j in itertools.combinations(seeds, 2)])) for c, cn in zip(cls, CN)}
    keys = list(phi)
    for k1, k2 in itertools.combinations(keys, 2):
        for c in cls:
            rbo_vals.append(d_rbo(imp(k1, c), imp(k2, c), P)); j_vals.append(1 - jaccard_star(imp(k1, c), imp(k2, c), 10))
    M["rbo_vs_jstar10_spearman"] = float(stats.spearmanr(rbo_vals, j_vals).statistic); M["n_pairs"] = len(rbo_vals)
    res["1_metric_check"] = M; log("1 done")

    # ---- 2. background ablation ----
    BG = {}
    for a in ["1", "0.1"]:
        same, client_vs_R = {cn: [] for cn in CN}, {cn: [] for cn in CN}
        for s in seeds:
            m = load_model(run(f"fedavg_dir_a{a}_s{s}"), X.shape[1], dev)
            tr, tb = load_partition(od, f"dir_a{a}_s{s}", dev, K)
            phiR_m = expected_gradients(m, X[R], Xbg, y[R], nsamples=NS, seed=0)
            g = torch.Generator().manual_seed(s)
            for k in range(K):
                lbg = X[tr[k][torch.randperm(len(tr[k]), generator=g)[:100].to(dev)]]
                yb = y[tb[k]]
                for c, cn in zip(cls, CN):
                    ic = tb[k][yb == c]
                    if len(ic) < MINC: continue
                    ic = ic[torch.randperm(len(ic), generator=g)[:PER].to(dev)]
                    p_sh = expected_gradients(m, X[ic], Xbg, y[ic], nsamples=NS, seed=k)
                    p_lo = expected_gradients(m, X[ic], lbg, y[ic], nsamples=NS, seed=k)
                    same[cn].append(client_eed(p_lo, p_sh, p=P, B=B, n_pairs=NP_, rng=rng)["EED"])
                    client_vs_R[cn].append(client_eed(p_lo, phiR_m[yR == c], p=P, B=B, n_pairs=NP_, rng=rng)["EED"])
            log(f"2 alpha={a} seed={s} done")
        BG[a] = {"local_vs_shared_same_samples": {cn: (float(np.mean(v)), len(v)) for cn, v in same.items() if v},
                 "client_local_bg_vs_R_shared_bg": {cn: (float(np.mean(v)), len(v)) for cn, v in client_vs_R.items() if v}}
    res["2_background"] = BG

    # ---- 3. KernelSHAP vs EG ----
    try:
        import shap
        m = load_model(run("central_s0"), X.shape[1], dev)
        g = torch.Generator().manual_seed(0)
        sel = torch.cat([R[torch.nonzero(y[R] == c).squeeze(1)[torch.randperm(int((yR == c).sum()), generator=g)[:10].to(dev)]] for c in cls])
        xs, ts = X[sel], y[sel].cpu().numpy()
        f = lambda a: m(torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dev)).detach().cpu().numpy()
        ke = shap.KernelExplainer(f, Xbg[:50].cpu().numpy())
        sv = ke.shap_values(xs.cpu().numpy(), nsamples=1000, silent=True)
        sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)          # (n, F, C)
        pk = sv[np.arange(len(ts)), :, ts]
        pe = expected_gradients(m, xs, Xbg, y[sel], nsamples=NS, seed=0)
        per = {}
        for c, cn in zip(cls, CN):
            mk = ts == c
            per[cn] = {"spearman": float(stats.spearmanr(importance(pk[mk]), importance(pe[mk])).statistic),
                       "D_rbo": float(d_rbo(importance(pk[mk]), importance(pe[mk]), P))}
        res["3_kernelshap_vs_eg"] = {"per_class": per, "overall_spearman": float(stats.spearmanr(importance(pk), importance(pe)).statistic),
                                     "n_samples": int(len(ts)), "kernel_nsamples": 1000}
        log("3 done")
    except ImportError:
        res["3_kernelshap_vs_eg"] = "skipped: pip install shap"

    json.dump(res, open("results/analysis/e7_ablations.json", "w"), indent=1, default=float)
    L = [f"=== 1. RBO vs 1-J*@10 agreement: Spearman {M['rbo_vs_jstar10_spearman']:.3f} over {M['n_pairs']} pairs ===",
         "1-J*@10 fidelity (fed vs central) | central baseline:"]
    for a in ["100", "1", "0.1"]:
        L.append(f"  a={a:4s} " + " ".join(f"{cn[:6]}:{M['fidelity_1minusJstar'][10][a][cn][0]:.2f}" for cn in CN))
    L.append("  base   " + " ".join(f"{cn[:6]}:{M['fidelity_1minusJstar'][10]['central_baseline'][cn]:.2f}" for cn in CN))
    L.append("\n=== 2. Background ablation (mean EED) ===")
    for a, d in BG.items():
        L.append(f"a={a}: local vs shared bg (same samples): " + " ".join(f"{cn[:6]}:{v[0]:.3f}" for cn, v in d["local_vs_shared_same_samples"].items()))
        L.append(f"       client(local bg) vs R(shared bg):  " + " ".join(f"{cn[:6]}:{v[0]:.3f}" for cn, v in d["client_local_bg_vs_R_shared_bg"].items()))
    k3 = res["3_kernelshap_vs_eg"]
    L.append("\n=== 3. KernelSHAP vs expected gradients ===")
    L.append(k3 if isinstance(k3, str) else f"overall Spearman {k3['overall_spearman']:.3f}; per class: " +
             " ".join(f"{cn[:6]}:{v['spearman']:.2f}" for cn, v in k3["per_class"].items()))
    open("results/analysis/e7_ablations_summary.txt", "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
