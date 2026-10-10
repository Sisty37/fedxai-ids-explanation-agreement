"""Run after train_local.py.
F. Heterogeneity effect relative to IID federated (alpha=100): per-class
   EED(fed_a_s, fed100_s'), s != s', minus the fed100 cross-seed baseline.
G. E4 2x2 factorized analysis (per class, NOT causal), alpha in {1, 0.1}:
   data = D(global|client, global|R); model = D(local|R, global|R);
   combined = D(local|client, global|R); interaction = combined - data - model.
   A class enters for a client only if the client has >= 30 TRAIN samples of it
   (a local model's explanation of a class it never saw is meaningless) and
   >= 30 Test-B samples. All terms noise-floor corrected.
Usage: python scripts/analyze_extra.py      Output: results/analysis/extra.json + extra_summary.txt
"""
import itertools, json, os, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, load_partition, CLASSES
from fedxai.train.model import MLP
from fedxai.explain.eg import expected_gradients
from fedxai.metrics.agreement import importance, d_rbo, bootstrap_importances, within_divergences, t_ci

NS, PER_CLASS = 1000, 200
C = len(CLASSES)


def main():
    cfg = yaml.safe_load(open("configs/protocol.yaml")); ex = cfg["explanation"]
    P, B, NP_, MINC = ex["rbo_p"], ex["bootstrap_B"], ex["noise_pairs"], ex["min_class_samples"]
    seeds, K = cfg["seeds"], cfg["partition"]["n_clients"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, idx, _ = load_2017(cfg["data"]["out_dir"], dev)
    R, bg = idx["R"], idx["background"]; yR = y[R].cpu().numpy(); Xbg = X[bg]
    cls = [c for c in range(C) if (yR == c).sum() >= MINC]
    rng = np.random.default_rng(0); t0 = time.time()
    log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {m}", flush=True)

    def load(path):
        m = MLP(X.shape[1], C).to(dev); m.load_state_dict(torch.load(path, map_location=dev, weights_only=True)); m.eval(); return m

    def within(phi):
        return within_divergences(bootstrap_importances(phi, B, rng), NP_, rng, P).mean()

    def eed(phi1, w1, phi2, w2):
        return d_rbo(importance(phi1), importance(phi2), P) - 0.5 * (w1 + w2)

    phiR, WR = {}, {}

    def prep(key, model):
        phiR[key] = expected_gradients(model, X[R], Xbg, y[R], nsamples=NS, seed=0)
        for c in cls: WR[(key, c)] = within(phiR[key][yR == c])

    def pe(k1, k2, c):
        m = yR == c
        return eed(phiR[k1][m], WR[(k1, c)], phiR[k2][m], WR[(k2, c)])

    run = lambda n: os.path.join("results", "runs", n, "final.pt")
    alphas = [str(a) for a in cfg["partition"]["dirichlet_alphas"]]
    for a in alphas:
        for s in seeds: prep(f"fed_{a}_s{s}", load(run(f"fedavg_dir_a{a}_s{s}")))
    log("explanations on R done")

    # ---- F. heterogeneity effect vs alpha = 100 ----
    base100 = {CLASSES[c]: float(np.mean([pe(f"fed_100_s{i}", f"fed_100_s{j}", c)
                                          for i, j in itertools.combinations(seeds, 2)])) for c in cls}
    F = {"fed100_seed_baseline": base100}
    for a in alphas:
        if a == "100": continue
        F[a] = {}
        for c in cls:
            v = [np.mean([pe(f"fed_{a}_s{s}", f"fed_100_s{s2}", c) for s2 in seeds if s2 != s]) - base100[CLASSES[c]]
                 for s in seeds]
            F[a][CLASSES[c]] = t_ci(v)
    log("F done")

    # ---- G. E4 factorized ----
    G = {}
    for a in ["1", "0.1"]:
        terms = {t: {CLASSES[c]: [] for c in cls} for t in ["data", "model", "combined", "interaction"]}
        for s in seeds:
            gk = f"fed_{a}_s{s}"; gm = load(run(f"fedavg_dir_a{a}_s{s}"))
            tr, tb = load_partition(cfg["data"]["out_dir"], f"dir_a{a}_s{s}", dev, K)
            seed_vals = {t: {CLASSES[c]: [] for c in cls} for t in terms}
            for k in range(K):
                lm = load(os.path.join("results", "local", f"dir_a{a}_s{s}_c{k}.pt"))
                lk = f"local_{a}_s{s}_c{k}"; prep(lk, lm)
                ytr, ytb = y[tr[k]], y[tb[k]]
                g = torch.Generator().manual_seed(1000 * s + k)
                for c in cls:
                    if (ytr == c).sum() < MINC or (ytb == c).sum() < MINC: continue
                    ic = tb[k][ytb == c]; ic = ic[torch.randperm(len(ic), generator=g)[:PER_CLASS].to(dev)]
                    pg = expected_gradients(gm, X[ic], Xbg, y[ic], nsamples=NS, seed=k)
                    pl = expected_gradients(lm, X[ic], Xbg, y[ic], nsamples=NS, seed=k)
                    wg, wl = within(pg), within(pl)
                    m = yR == c
                    d = eed(pg, wg, phiR[gk][m], WR[(gk, c)])
                    mo = pe(lk, gk, c)
                    co = eed(pl, wl, phiR[gk][m], WR[(gk, c)])
                    for t, v in zip(["data", "model", "combined", "interaction"], [d, mo, co, co - d - mo]):
                        seed_vals[t][CLASSES[c]].append(v)
            for t in terms:
                for cn, v in seed_vals[t].items():
                    if v: terms[t][cn].append(float(np.mean(v)))
            log(f"G alpha={a} seed={s} done")
        G[a] = {t: {cn: (t_ci(v) if len(v) >= 2 else None, len(v)) for cn, v in d.items()} for t, d in terms.items()}
    res = {"F_heterogeneity_vs_alpha100": F, "G_factorized": G}
    os.makedirs("results/analysis", exist_ok=True)
    json.dump(res, open("results/analysis/extra.json", "w"), indent=1, default=float)
    L = ["=== F. Heterogeneity effect: EED vs fed alpha=100, minus fed100 seed baseline ==="]
    L.append("alpha   " + " ".join(f"{CLASSES[c][:9]:>12s}" for c in cls))
    for a in alphas[1:]:
        L.append(f"{a:7s} " + " ".join(f"{F[a][CLASSES[c]][0]:6.3f}+-{F[a][CLASSES[c]][1]:.3f}" for c in cls))
    for a in ["1", "0.1"]:
        L.append(f"\n=== G. Factorized, alpha={a} (mean over seeds; n seeds with data) ===")
        for t in ["data", "model", "combined", "interaction"]:
            L.append(f"{t:12s} " + " ".join(
                f"{CLASSES[c][:6]}:{(G[a][t][CLASSES[c]][0][0] if G[a][t][CLASSES[c]][0] else float('nan')):6.3f}(n={G[a][t][CLASSES[c]][1]})"
                for c in cls))
    open("results/analysis/extra_summary.txt", "w").write("\n".join(L)); print("\n".join(L)); log("all done")


if __name__ == "__main__":
    main()
