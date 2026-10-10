"""Second-architecture validation (1D-CNN), compact design from the review checklist.

Train (resumable):  centralized, FedAvg alpha in {100, 1, 0.1}, FedProx (mu=0.01) alpha=100,
                    5 seeds -> 25 runs, 50 rounds/epochs each (~6-8 h on RTX 4050).
Analyse:            the paper's main findings re-measured for the CNN:
                    detection, seed baseline, FedAvg-vs-central fidelity (alpha=100),
                    heterogeneity drift (alpha=1, 0.1 vs 100), class-mix effect,
                    SHAP-background effect (alpha=1), FedProx-minus-FedAvg fidelity (alpha=100).

Usage:  python scripts/cnn_validation.py --dry-run
        python scripts/cnn_validation.py            # train missing runs, then analyse
        python scripts/cnn_validation.py --analyze  # analysis only
Output: results/runs/cnn_*/ , results/analysis/cnn_validation.json + cnn_validation_summary.txt
"""
import argparse, itertools, json, os, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, load_partition, class_weights, CLASSES
from fedxai.train.model import build_model
from fedxai.train.fl import train_centralized, train_federated
from fedxai.train.evaluate import detection_metrics
from functools import partial
from fedxai.explain.eg import expected_gradients as _eg
# CNN windows are memory-heavy: explain 8 flows at a time (8 x 1000 samples) to fit a 6 GB GPU
expected_gradients = partial(_eg, chunk=8)
from fedxai.metrics.agreement import (importance, d_rbo, bootstrap_importances, within_divergences,
                                      client_eed, t_ci)

ARCH, ROUNDS, NS, PER = "cnn", 50, 1000, 200
C = len(CLASSES)


def runs(seeds):
    out = []
    for s in seeds:
        out.append(("central", None, 0.0, s))
        for a in ["100", "1", "0.1"]:
            out.append(("fedavg", a, 0.0, s))
        out.append(("fedprox", "100", 0.01, s))
    return out


def name(kind, a, mu, s):
    if kind == "central": return f"cnn_central_s{s}"
    if kind == "fedavg": return f"cnn_fedavg_dir_a{a}_s{s}"
    return f"cnn_fedprox_mu{mu}_dir_a{a}_s{s}"


def rdir(n): return os.path.join("results", "runs", n)


def train(cfg, seeds, X, y, idx, dev):
    w = class_weights(y[idx["train"]]); K = cfg["partition"]["n_clients"]
    todo = [r for r in runs(seeds) if not os.path.exists(os.path.join(rdir(name(*r)), "metrics.json"))]
    t0 = time.time()
    for i, (kind, a, mu, s) in enumerate(todo, 1):
        n = name(kind, a, mu, s); os.makedirs(rdir(n), exist_ok=True)
        print(f"\n=== [{i}/{len(todo)}] {n} ===", flush=True)
        torch.manual_seed(s); np.random.seed(s)
        m = build_model(ARCH, X.shape[1], C).to(dev)
        lg = lambda msg: print(msg, flush=True) if any(f" {e}/50 " in msg + " " for e in (1, 10, 25, 50)) else None
        if kind == "central":
            train_centralized(m, X, y, idx["train"], ROUNDS, w, seed=s, log=lg)
        else:
            tr, _ = load_partition(cfg["data"]["out_dir"], f"dir_a{a}_s{s}", dev, K)
            train_federated(m, X, y, tr, ROUNDS, w, mu=mu, seed=s, log=lg)
        torch.save(m.state_dict(), os.path.join(rdir(n), "final.pt"))
        json.dump({"run": n, "arch": ARCH, "seed": s, "mu": mu,
                   "testb": detection_metrics(m, X, y, idx["test_b"], C)}, open(os.path.join(rdir(n), "metrics.json"), "w"))
        el = time.time() - t0
        print(f"done | elapsed {el / 3600:.2f} h | ETA {(len(todo) - i) * el / i / 3600:.2f} h", flush=True)


def analyse(cfg, seeds, X, y, idx, dev):
    ex = cfg["explanation"]; P, B, NP_, MINC = ex["rbo_p"], ex["bootstrap_B"], ex["noise_pairs"], ex["min_class_samples"]
    K = cfg["partition"]["n_clients"]; od = cfg["data"]["out_dir"]
    R, bg = idx["R"], idx["background"]; yR = y[R].cpu().numpy(); Xbg = X[bg]
    cls = [c for c in range(C) if (yR == c).sum() >= MINC]; CN = [CLASSES[c] for c in cls]
    rng = np.random.default_rng(0)
    def load(n):
        m = build_model(ARCH, X.shape[1], C).to(dev)
        m.load_state_dict(torch.load(os.path.join(rdir(n), "final.pt"), map_location=dev, weights_only=True))
        m.eval()
        return m
    phiR, W, models = {}, {}, {}
    for r in runs(seeds):
        n = name(*r); models[n] = load(n)
        phiR[n] = expected_gradients(models[n], X[R], Xbg, y[R], nsamples=NS, seed=0)
        torch.cuda.empty_cache()
        for c in cls:
            W[(n, c)] = within_divergences(bootstrap_importances(phiR[n][yR == c], B, rng), NP_, rng, P).mean()
    print("explanations done", flush=True)

    def pe(n1, n2, c):
        m = yR == c
        return d_rbo(importance(phiR[n1][m]), importance(phiR[n2][m]), P) - 0.5 * (W[(n1, c)] + W[(n2, c)])

    cen = lambda s: name("central", None, 0.0, s)
    fa = lambda a, s: name("fedavg", a, 0.0, s)
    res = {"arch": ARCH, "seeds": seeds}
    res["detection"] = {k: t_ci([json.load(open(os.path.join(rdir(f(s)), "metrics.json")))["testb"]["macro_f1"] for s in seeds])
                        for k, f in [("central", cen), ("fedavg_a100", lambda s: fa("100", s)), ("fedavg_a1", lambda s: fa("1", s)),
                                     ("fedavg_a0.1", lambda s: fa("0.1", s)), ("fedprox_a100", lambda s: name("fedprox", "100", 0.01, s))]}
    base = {c: np.mean([pe(cen(i), cen(j), c) for i, j in itertools.combinations(seeds, 2)]) for c in cls}
    res["seed_baseline"] = {CLASSES[c]: float(base[c]) for c in cls}
    fid = lambda n, s, c: np.mean([pe(n, cen(s2), c) for s2 in seeds if s2 != s]) - base[c]
    res["fidelity_fedavg_a100"] = {CLASSES[c]: t_ci([fid(fa("100", s), s, c) for s in seeds]) for c in cls}
    res["fedprox_minus_fedavg_a100"] = {CLASSES[c]: t_ci([fid(name("fedprox", "100", 0.01, s), s, c) - fid(fa("100", s), s, c) for s in seeds]) for c in cls}
    b100 = {c: np.mean([pe(fa("100", i), fa("100", j), c) for i, j in itertools.combinations(seeds, 2)]) for c in cls}
    res["heterogeneity_drift"] = {}
    for a in ["1", "0.1"]:
        per = {c: [np.mean([pe(fa(a, s), fa("100", s2), c) for s2 in seeds if s2 != s]) - b100[c] for s in seeds] for c in cls}
        res["heterogeneity_drift"][a] = {"per_class": {CLASSES[c]: t_ci(v) for c, v in per.items()},
                                         "mean_over_classes": t_ci([np.mean([per[c][i] for c in cls]) for i in range(len(seeds))])}
    print("fidelity/heterogeneity done", flush=True)
    mix = {}
    for a in ["1", "0.1"]:
        dR, dM = [], []
        for s in seeds:
            n = fa(a, s); m = models[n]
            _, tb = load_partition(od, f"dir_a{a}_s{s}", dev, K)
            means = {c: np.abs(phiR[n][yR == c]).mean(0) for c in range(C) if (yR == c).any()}
            sR = importance(phiR[n]); vr, vm = [], []
            for k, ib in enumerate(tb):
                g = torch.Generator().manual_seed(1000 * s + k)
                sel = ib[torch.randperm(len(ib), generator=g)[:ex["max_explained_per_client"]].to(dev)]
                ph = expected_gradients(m, X[sel], Xbg, y[sel], nsamples=NS, seed=k)
                pi = np.bincount(y[sel].cpu().numpy(), minlength=C) / len(sel)
                pred = sum(pi[c] * means[c] for c in means)
                vr.append(d_rbo(importance(ph), sR, P)); vm.append(d_rbo(importance(ph), pred / pred.sum(), P))
            dR.append(np.mean(vr)); dM.append(np.mean(vm))
        mix[a] = {"D_client_vs_R": t_ci(dR), "D_client_vs_mixture": t_ci(dM), "share_explained_by_mix": float(1 - np.mean(dM) / np.mean(dR))}
    res["class_mix"] = mix
    bgres = {cn: [] for cn in CN}
    for s in seeds:
        n = fa("1", s); m = models[n]
        tr, tb = load_partition(od, f"dir_a1_s{s}", dev, K)
        g = torch.Generator().manual_seed(s)
        for k in range(K):
            lbg = X[tr[k][torch.randperm(len(tr[k]), generator=g)[:100].to(dev)]]; yb = y[tb[k]]
            for c, cn in zip(cls, CN):
                ic = tb[k][yb == c]
                if len(ic) < MINC: continue
                ic = ic[torch.randperm(len(ic), generator=g)[:PER].to(dev)]
                p_sh = expected_gradients(m, X[ic], Xbg, y[ic], nsamples=NS, seed=k)
                p_lo = expected_gradients(m, X[ic], lbg, y[ic], nsamples=NS, seed=k)
                bgres[cn].append(client_eed(p_lo, p_sh, p=P, B=B, n_pairs=NP_, rng=rng)["EED"])
    res["background_effect_a1"] = {cn: (float(np.mean(v)), len(v)) for cn, v in bgres.items() if v}
    os.makedirs("results/analysis", exist_ok=True)
    json.dump(res, open("results/analysis/cnn_validation.json", "w"), indent=1, default=float)
    L = [f"=== 1D-CNN validation ({len(seeds)} seeds) ===", "Detection macro-F1:"]
    L += [f"  {k:13s} {v[0]:.4f} +- {v[1]:.4f}" for k, v in res["detection"].items()]
    L.append("Seed baseline: " + " ".join(f"{k[:6]}:{v:.3f}" for k, v in res["seed_baseline"].items()))
    L.append("FedAvg a=100 fidelity excess: " + " ".join(f"{k[:6]}:{v[0]:.3f}" for k, v in res["fidelity_fedavg_a100"].items()))
    for a, d in res["heterogeneity_drift"].items():
        L.append(f"Heterogeneity drift a={a}: mean {d['mean_over_classes'][0]:.3f} +- {d['mean_over_classes'][1]:.3f}")
    for a, d in mix.items():
        L.append(f"Class mix a={a}: D_vs_R {d['D_client_vs_R'][0]:.3f}, D_vs_mix {d['D_client_vs_mixture'][0]:.3f}, share {d['share_explained_by_mix']:.2f}")
    L.append("Background effect a=1: " + " ".join(f"{k[:6]}:{v[0]:.3f}" for k, v in res["background_effect_a1"].items()))
    L.append("FedProx - FedAvg fidelity a=100: " + " ".join(f"{k[:6]}:{v[0]:+.3f}" for k, v in res["fedprox_minus_fedavg_a100"].items()))
    open("results/analysis/cnn_validation_summary.txt", "w").write("\n".join(L)); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/protocol.yaml"))
    todo = [name(*r) for r in runs(a.seeds) if not os.path.exists(os.path.join(rdir(name(*r)), "metrics.json"))]
    print(f"{len(runs(a.seeds))} CNN runs planned, {len(todo)} to train")
    if a.dry_run:
        for n in todo: print("  ", n)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, idx, _ = load_2017(cfg["data"]["out_dir"], dev)
    if not a.analyze: train(cfg, a.seeds, X, y, idx, dev)
    analyse(cfg, a.seeds, X, y, idx, dev)


if __name__ == "__main__":
    main()
