"""Analysis of E1-E3 from saved checkpoints (no retraining).

A. Seed baseline      B_c: per-class EED between centralized models of different seeds.
B. Fidelity           per-class EED(fed_part_s, central_s'), s' != s, minus B_c.
C. Detection          macro-F1 and per-class F1 per partition (mean +- t-CI, 5 seeds).
D. Mixture effect     same global model: client overall D vs R, and vs class-mix prediction;
                      per-class client EED vs R (expected ~0 under label skew).
E. Trajectory         alpha in {100, 1, 0.1}: macro-F1 and mean fidelity EED per round.

Usage: python scripts/analyze_main.py          (~20-40 min on RTX 4050)
Output: results/analysis/analysis.json, results/analysis/summary.txt
"""
import itertools, json, os, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, load_partition, CLASSES
from fedxai.train.model import MLP
from fedxai.train.evaluate import detection_metrics
from fedxai.explain.eg import expected_gradients
from fedxai.metrics.agreement import (importance, d_rbo, bootstrap_importances,
                                      within_divergences, client_eed, t_ci)

NS = 1000                      # expected-gradients samples (locked after pilot)
PER_CLASS_CLIENT = 200         # explained samples per class per client (D)
TRAJ_ALPHAS, TRAJ_ROUNDS = ["100", "1", "0.1"], [1, 5, 10, 20, 30, 40, 50]
C = len(CLASSES)


def main():
    cfg = yaml.safe_load(open("configs/protocol.yaml"))
    ex = cfg["explanation"]; P, B, NP_, MINC = ex["rbo_p"], ex["bootstrap_B"], ex["noise_pairs"], ex["min_class_samples"]
    seeds = cfg["seeds"]
    parts = [f"dir_a{a}" for a in cfg["partition"]["dirichlet_alphas"]] + ["attack_skew"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join("results", "analysis"); os.makedirs(out, exist_ok=True)
    X, y, idx, _ = load_2017(cfg["data"]["out_dir"], dev)
    R, bg = idx["R"], idx["background"]; yR = y[R].cpu().numpy(); Xbg = X[bg]
    classes_R = [c for c in range(C) if (yR == c).sum() >= MINC]
    rng = np.random.default_rng(0)
    t0 = time.time()
    log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {m}", flush=True)

    def load(path):
        m = MLP(X.shape[1], C).to(dev)
        m.load_state_dict(torch.load(path, map_location=dev, weights_only=True)); m.eval()
        return m

    phiR, W = {}, {}

    def prep(key, path):
        """EG on R + within-bootstrap divergences per class (cached)."""
        if key in phiR: return
        m = load(path)
        phiR[key] = expected_gradients(m, X[R], Xbg, y[R], nsamples=NS, seed=0)
        for c in classes_R:
            W[(key, c)] = within_divergences(bootstrap_importances(phiR[key][yR == c], B, rng), NP_, rng, P)

    def pair_eed(k1, k2, c):
        msk = yR == c
        D = d_rbo(importance(phiR[k1][msk]), importance(phiR[k2][msk]), P)
        return D - 0.5 * (W[(k1, c)].mean() + W[(k2, c)].mean())

    run = lambda name: os.path.join("results", "runs", name)
    res = {"classes_analyzed": [CLASSES[c] for c in classes_R], "nsamples": NS}

    # ---- A. seed baseline ----
    for s in seeds: prep(f"central_s{s}", os.path.join(run(f"central_s{s}"), "final.pt"))
    base = {}
    for c in classes_R:
        v = [pair_eed(f"central_s{a}", f"central_s{b}", c) for a, b in itertools.combinations(seeds, 2)]
        base[CLASSES[c]] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
    res["A_seed_baseline"] = base
    log("A done")

    # ---- B + C. fidelity and detection per partition ----
    fid, det = {}, {}
    for part in parts:
        per_seed = {CLASSES[c]: [] for c in classes_R}
        f1s, pcf1 = [], []
        for s in seeds:
            k = f"fed_{part}_s{s}"
            prep(k, os.path.join(run(f"fedavg_{part}_s{s}"), "final.pt"))
            for c in classes_R:
                per_seed[CLASSES[c]].append(np.mean([pair_eed(k, f"central_s{s2}", c) for s2 in seeds if s2 != s]))
            mj = json.load(open(os.path.join(run(f"fedavg_{part}_s{s}"), "metrics.json")))
            f1s.append(mj["testb"]["macro_f1"]); pcf1.append(mj["testb"]["per_class_f1"])
        fid[part] = {}
        for cn, v in per_seed.items():
            m, h = t_ci(v); ex_v = [x - base[cn]["mean"] for x in v]; me, he = t_ci(ex_v)
            fid[part][cn] = {"EED": m, "ci": h, "excess_over_seed_baseline": me, "excess_ci": he}
        m, h = t_ci(f1s); pc = np.array(pcf1)
        det[part] = {"macro_f1": m, "ci": h,
                     "per_class_f1": {CLASSES[c]: t_ci(pc[:, c]) for c in range(C)}}
        log(f"B/C {part} done")
    cf = [json.load(open(os.path.join(run(f"central_s{s}"), "metrics.json")))["testb"] for s in seeds]
    m, h = t_ci([d["macro_f1"] for d in cf]); pc = np.array([d["per_class_f1"] for d in cf])
    det["central"] = {"macro_f1": m, "ci": h, "per_class_f1": {CLASSES[c]: t_ci(pc[:, c]) for c in range(C)}}
    res["B_fidelity"], res["C_detection"] = fid, det

    # ---- D. mixture effect + per-class client agreement (same global model) ----
    mixres = {}
    for part in parts:
        dR, dMix, pce = [], [], {CLASSES[c]: [] for c in classes_R}
        for s in seeds:
            k = f"fed_{part}_s{s}"; m = load(os.path.join(run(f"fedavg_{part}_s{s}"), "final.pt"))
            _, tb = load_partition(cfg["data"]["out_dir"], f"{part}_s{s}", dev, cfg["partition"]["n_clients"])
            means = {c: np.abs(phiR[k][yR == c]).mean(0) for c in range(C) if (yR == c).any()}
            sR = importance(phiR[k])
            for ci, ib in enumerate(tb):
                g = torch.Generator().manual_seed(1000 * s + ci)
                sel = ib[torch.randperm(len(ib), generator=g)[:ex["max_explained_per_client"]].to(dev)]
                phi = expected_gradients(m, X[sel], Xbg, y[sel], nsamples=NS, seed=ci)
                yk = y[sel].cpu().numpy(); pi = np.bincount(yk, minlength=C) / len(yk)
                pred = sum(pi[c] * means[c] for c in means)
                sk = importance(phi)
                dR.append(d_rbo(sk, sR, P)); dMix.append(d_rbo(sk, pred / pred.sum(), P))
                yb = y[ib]
                for c in classes_R:
                    ic = ib[yb == c]
                    if len(ic) < MINC: continue
                    ic = ic[torch.randperm(len(ic), generator=g)[:PER_CLASS_CLIENT].to(dev)]
                    phc = expected_gradients(m, X[ic], Xbg, y[ic], nsamples=NS, seed=ci)
                    r = client_eed(phc, phiR[k][yR == c], p=P, B=B, n_pairs=NP_, rng=rng, ref_within=W[(k, c)])
                    pce[CLASSES[c]].append((r["EED"], r["significant"]))
        mixres[part] = {"D_client_vs_R": t_ci(dR), "D_client_vs_mixture": t_ci(dMix),
                        "share_explained_by_mix": 1 - float(np.mean(dMix) / np.mean(dR)),
                        "per_class_client_EED": {cn: {"mean": float(np.mean([e for e, _ in v])) if v else None,
                                                      "sig_rate": float(np.mean([g_ for _, g_ in v])) if v else None,
                                                      "n_client_cells": len(v)} for cn, v in pce.items()}}
        log(f"D {part} done")
    res["D_mixture"] = mixres

    # ---- E. trajectory ----
    traj = {}
    for a in TRAJ_ALPHAS:
        traj[a] = {}
        for r in TRAJ_ROUNDS:
            f1, fe = [], []
            for s in seeds:
                path = os.path.join(run(f"fedavg_dir_a{a}_s{s}"), f"round_{r}.pt")
                k = f"traj_a{a}_r{r}_s{s}"; prep(k, path)
                f1.append(detection_metrics(load(path), X, y, idx["test_b"], C)["macro_f1"])
                fe.append(np.mean([np.mean([pair_eed(k, f"central_s{s2}", c) for s2 in seeds if s2 != s])
                                   - base[CLASSES[c]]["mean"] for c in classes_R]))
            traj[a][r] = {"macro_f1": t_ci(f1), "fidelity_excess_mean_over_classes": t_ci(fe)}
        log(f"E alpha={a} done")
    res["E_trajectory"] = traj

    json.dump(res, open(os.path.join(out, "analysis.json"), "w"), indent=1, default=float)
    lines = ["=== Detection (macro-F1, mean +- 95% t-CI, 5 seeds) ==="]
    for p_, d in det.items(): lines.append(f"{p_:14s} {d['macro_f1']:.4f} +- {d['ci']:.4f}")
    lines.append("\n=== Seed baseline B_c (central vs central) ===")
    for cn, b in base.items(): lines.append(f"{cn:11s} {b['mean']:.4f} (sd {b['sd']:.4f})")
    lines.append("\n=== Fidelity excess over seed baseline (mean over seeds) ===")
    lines.append("partition      " + " ".join(f"{cn[:9]:>9s}" for cn in base))
    for p_ in parts:
        lines.append(f"{p_:14s} " + " ".join(f"{fid[p_][cn]['excess_over_seed_baseline']:9.4f}" for cn in base))
    lines.append("\n=== Mixture effect (same global model) ===")
    for p_ in parts:
        mr = mixres[p_]
        lines.append(f"{p_:14s} D_vs_R {mr['D_client_vs_R'][0]:.4f}  D_vs_mix {mr['D_client_vs_mixture'][0]:.4f}  "
                     f"mix share {mr['share_explained_by_mix']:.2f}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines))
    print("\n".join(lines)); log("all done")


if __name__ == "__main__":
    main()
