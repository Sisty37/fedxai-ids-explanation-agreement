"""Run after the FedProx runs (train_all.py --fedprox).
H. FedProx vs FedAvg (RQ4a): detection, and per-class fidelity to the centralized
   reference (EED vs central_s', s' != s, minus central seed baseline). Paired by seed:
   diff = prox - avg (negative = FedProx closer to centralized). Plus mu sweep at alpha=0.1.
I. Detection-explanation coupling (RQ2), FedAvg runs (6 alphas + attack-skew, 5 seeds):
   per run: detection drop (central mean F1 - run F1) and explanation drift
   (EED vs fed alpha=100 runs minus fed100 seed baseline), overall and per class.
   Spearman across runs (bootstrap 95% CI), pooled and within-alpha (alpha means removed),
   and the first alpha where each signal is significantly > 0 (onset).
Usage: python scripts/analyze_fedprox_rq2.py   Output: results/analysis/fedprox_rq2.json + _summary.txt
"""
import itertools, json, os, sys, time
import numpy as np, torch, yaml
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, CLASSES
from fedxai.train.model import MLP
from fedxai.explain.eg import expected_gradients
from fedxai.metrics.agreement import importance, d_rbo, bootstrap_importances, within_divergences, t_ci

NS = 1000
C = len(CLASSES)


def spearman_ci(x, y, n_boot=2000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            bs.append(stats.spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"rho": float(rho), "ci": [float(lo), float(hi)], "n": int(len(x))}


def main():
    cfg = yaml.safe_load(open("configs/protocol.yaml")); ex = cfg["explanation"]
    P, B, NP_, MINC = ex["rbo_p"], ex["bootstrap_B"], ex["noise_pairs"], ex["min_class_samples"]
    seeds = cfg["seeds"]; alphas = [str(a) for a in cfg["partition"]["dirichlet_alphas"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, idx, _ = load_2017(cfg["data"]["out_dir"], dev)
    R, bg = idx["R"], idx["background"]; yR = y[R].cpu().numpy(); Xbg = X[bg]
    cls = [c for c in range(C) if (yR == c).sum() >= MINC]; CN = [CLASSES[c] for c in cls]
    rng = np.random.default_rng(0); t0 = time.time()
    log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {m}", flush=True)
    run = lambda n: os.path.join("results", "runs", n)
    phiR, WR = {}, {}

    def prep(key, name):
        if key in phiR: return
        m = MLP(X.shape[1], C).to(dev)
        m.load_state_dict(torch.load(os.path.join(run(name), "final.pt"), map_location=dev, weights_only=True)); m.eval()
        phiR[key] = expected_gradients(m, X[R], Xbg, y[R], nsamples=NS, seed=0)
        for c in cls:
            WR[(key, c)] = within_divergences(bootstrap_importances(phiR[key][yR == c], B, rng), NP_, rng, P).mean()

    def pe(k1, k2, c):
        m = yR == c
        return d_rbo(importance(phiR[k1][m]), importance(phiR[k2][m]), P) - 0.5 * (WR[(k1, c)] + WR[(k2, c)])

    f1 = lambda name: json.load(open(os.path.join(run(name), "metrics.json")))["testb"]
    for s in seeds: prep(f"cen{s}", f"central_s{s}")
    parts = alphas + ["skew"]
    pname = lambda p, s: f"fedavg_attack_skew_s{s}" if p == "skew" else f"fedavg_dir_a{p}_s{s}"
    for p in parts:
        for s in seeds: prep(f"avg_{p}_{s}", pname(p, s))
    for a in alphas:
        for s in seeds: prep(f"prox0.01_{a}_{s}", f"fedprox_mu0.01_dir_a{a}_s{s}")
    for mu in ["0.001", "0.1"]:
        for s in seeds: prep(f"prox{mu}_0.1_{s}", f"fedprox_mu{mu}_dir_a0.1_s{s}")
    log("explanations done")

    cbase = {c: np.mean([pe(f"cen{i}", f"cen{j}", c) for i, j in itertools.combinations(seeds, 2)]) for c in cls}
    fid = lambda k, s, c: np.mean([pe(k, f"cen{s2}", c) for s2 in seeds if s2 != s]) - cbase[c]

    # ---- H. FedProx vs FedAvg ----
    H = {"detection": {}, "fidelity": {}, "paired_diff_prox_minus_avg": {}, "mu_sweep_alpha0.1": {}}
    cen_f1 = [f1(f"central_s{s}")["macro_f1"] for s in seeds]
    for a in alphas:
        av = [f1(pname(a, s))["macro_f1"] for s in seeds]
        pr = [f1(f"fedprox_mu0.01_dir_a{a}_s{s}")["macro_f1"] for s in seeds]
        H["detection"][a] = {"fedavg": t_ci(av), "fedprox": t_ci(pr), "paired_diff": t_ci(np.subtract(pr, av))}
        H["fidelity"][a], H["paired_diff_prox_minus_avg"][a] = {}, {}
        for c, cn in zip(cls, CN):
            fa = [fid(f"avg_{a}_{s}", s, c) for s in seeds]; fp = [fid(f"prox0.01_{a}_{s}", s, c) for s in seeds]
            H["fidelity"][a][cn] = {"fedavg": t_ci(fa), "fedprox": t_ci(fp)}
            H["paired_diff_prox_minus_avg"][a][cn] = t_ci(np.subtract(fp, fa))
    for mu in ["0.001", "0.01", "0.1"]:
        name = lambda s: f"fedprox_mu{mu}_dir_a0.1_s{s}"
        H["mu_sweep_alpha0.1"][mu] = {"macro_f1": t_ci([f1(name(s))["macro_f1"] for s in seeds]),
                                      "fidelity": {cn: t_ci([fid(f"prox{mu}_0.1_{s}", s, c) for s in seeds]) for c, cn in zip(cls, CN)}}
    log("H done")

    # ---- I. coupling (FedAvg) ----
    b100 = {c: np.mean([pe(f"avg_100_{i}", f"avg_100_{j}", c) for i, j in itertools.combinations(seeds, 2)]) for c in cls}
    cen_pc = np.mean([f1(f"central_s{s}")["per_class_f1"] for s in seeds], axis=0)
    rows = []
    for p in parts:
        for s in seeds:
            k = f"avg_{p}_{s}"; m = f1(pname(p, s))
            drift = {c: (np.mean([pe(k, f"avg_100_{s2}", c) for s2 in seeds if s2 != s]) if p != "100"
                         else np.mean([pe(k, f"avg_100_{s2}", c) for s2 in seeds if s2 != s])) - b100[c] for c in cls}
            rows.append({"part": p, "seed": s, "f1_drop": float(np.mean(cen_f1) - m["macro_f1"]),
                         "drift": float(np.mean(list(drift.values()))),
                         "per_class": {CLASSES[c]: {"f1_drop": float(cen_pc[c] - m["per_class_f1"][c]), "drift": float(drift[c])} for c in cls}})
    x = [r["f1_drop"] for r in rows]; yv = [r["drift"] for r in rows]
    I = {"runs": rows, "pooled_spearman": spearman_ci(x, yv)}
    # within-alpha: remove partition means
    xr, yr = [], []
    for p in parts:
        sub = [r for r in rows if r["part"] == p]
        mx, my = np.mean([r["f1_drop"] for r in sub]), np.mean([r["drift"] for r in sub])
        xr += [r["f1_drop"] - mx for r in sub]; yr += [r["drift"] - my for r in sub]
    I["within_alpha_spearman"] = spearman_ci(xr, yr)
    cx = [r["per_class"][cn]["f1_drop"] for r in rows for cn in CN]
    cy = [r["per_class"][cn]["drift"] for r in rows for cn in CN]
    I["per_class_cells_spearman"] = spearman_ci(cx, cy)
    onset = {}
    for p in alphas:
        sub = [r for r in rows if r["part"] == p]
        dm, dh = t_ci([r["f1_drop"] - np.mean([q["f1_drop"] for q in rows if q["part"] == "100"]) for r in sub])
        em, eh = t_ci([r["drift"] for r in sub])
        onset[p] = {"detection_drop_vs_100": [dm, dh, bool(dm - dh > 0)], "explanation_drift": [em, eh, bool(em - eh > 0)]}
    I["onset"] = onset
    log("I done")

    out = {"H_fedprox": H, "I_coupling": I}
    json.dump(out, open("results/analysis/fedprox_rq2.json", "w"), indent=1, default=float)
    L = ["=== H. Detection macro-F1: FedAvg | FedProx(mu=0.01) | paired diff ==="]
    for a in alphas:
        d = H["detection"][a]
        L.append(f"a={a:5s} {d['fedavg'][0]:.4f}+-{d['fedavg'][1]:.4f} | {d['fedprox'][0]:.4f}+-{d['fedprox'][1]:.4f} | {d['paired_diff'][0]:+.4f}+-{d['paired_diff'][1]:.4f}")
    L.append("\n=== H. Fidelity paired diff (FedProx - FedAvg; negative = closer to centralized) ===")
    L.append("alpha  " + " ".join(f"{cn[:9]:>15s}" for cn in CN))
    for a in alphas:
        L.append(f"{a:6s} " + " ".join(f"{H['paired_diff_prox_minus_avg'][a][cn][0]:+.3f}+-{H['paired_diff_prox_minus_avg'][a][cn][1]:.3f}" for cn in CN))
    L.append("\n=== H. mu sweep at alpha=0.1: macro-F1 | mean fidelity excess over classes ===")
    for mu, d in H["mu_sweep_alpha0.1"].items():
        L.append(f"mu={mu:6s} {d['macro_f1'][0]:.4f}+-{d['macro_f1'][1]:.4f} | {np.mean([d['fidelity'][cn][0] for cn in CN]):.3f}")
    L.append("\n=== I. Coupling: detection drop vs explanation drift (Spearman, bootstrap 95% CI) ===")
    for k in ["pooled_spearman", "within_alpha_spearman", "per_class_cells_spearman"]:
        v = I[k]; L.append(f"{k:26s} rho={v['rho']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] n={v['n']}")
    L.append("\n=== I. Onset (significant > 0?): detection drop vs alpha=100 | explanation drift ===")
    for a, d in onset.items():
        L.append(f"a={a:5s} det {d['detection_drop_vs_100'][0]:+.4f}+-{d['detection_drop_vs_100'][1]:.4f} {'SIG' if d['detection_drop_vs_100'][2] else '-  '} | "
                 f"expl {d['explanation_drift'][0]:+.4f}+-{d['explanation_drift'][1]:.4f} {'SIG' if d['explanation_drift'][2] else '-'}")
    open("results/analysis/fedprox_rq2_summary.txt", "w").write("\n".join(L)); print("\n".join(L)); log("all done")


if __name__ == "__main__":
    main()
