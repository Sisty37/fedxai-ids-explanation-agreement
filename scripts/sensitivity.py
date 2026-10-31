"""Camera-ready review items (R4, R8): sensitivity and robustness of EED on CICIDS2017.
Uses the saved checkpoints only (no retraining). Resumable: SHAP matrices are cached
in results/cache/sens/, so a crash or Ctrl+C loses at most one model.

S1  RBO weight p in {0.8, 0.9, 0.95, 0.98} and a Spearman divergence: seed baseline,
    federation excess (alpha=100), heterogeneity drift (alpha=0.1) and the class-mix
    share (alpha in {1, 0.1}).
S2  Shared background size m in {25, 50, 100, 200}, plus two more random draws of a
    100-flow background (same model, different shared background).
S3  Number of explained flows n in {10, 20, 30, 50, 100}: noise floor and EED between
    two centralized seeds on random subsets; how many client-class cells pass the
    minimum-flow threshold T in {10, 30, 50, 100}.
S4  Bootstrap configuration: resamples B in {200, 500, 1000, 2000} x pairs in
    {1000, 5000, 20000}, ten repetitions -> Monte Carlo uncertainty of the floor.
S5  KernelSHAP per class (20 flows per class, 15 models): seed baseline, federation
    excess and drift computed with KernelSHAP and with expected gradients on the
    same flows (needs `pip install shap`; skipped otherwise).
S6  Cost: time per expected-gradients run and per noise floor; size of the per-class
    summary a client would send.
S7  Per-class test counts and per-class F1 (central, FedAvg alpha 100 / 0.1,
    attack-skew, FedProx mu=0.01 alpha 0.1), from metrics.json.

Usage:  python scripts/sensitivity.py            (~1-2 h on an RTX 4050)
        python scripts/sensitivity.py --only S7  (run selected parts, e.g. S1,S7)
Output: results/analysis/sensitivity.json + sensitivity_summary.txt
"""
import argparse, itertools, json, os, sys, time
import numpy as np, torch, yaml
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, load_partition, CLASSES
from fedxai.train.common import load_model
from fedxai.explain.eg import expected_gradients
from fedxai.metrics.agreement import importance, rbo_ext_batch, bootstrap_importances, t_ci

NS = int(os.environ.get("FEDXAI_NS", 1000))   # expected-gradients samples (main protocol: 1000; env override only for smoke tests)
PS = [0.8, 0.9, 0.95, 0.98]
C = len(CLASSES)
CACHE = os.path.join("results", "cache", "sens")


# ---------------------------------------------------------------- helpers
def d_sp_batch(a, b):
    """Spearman divergence (1 - rho) / 2 for rows of a and b."""
    ra = np.argsort(np.argsort(-np.atleast_2d(a), axis=1, kind="stable"), axis=1).astype(float)
    rb = np.argsort(np.argsort(-np.atleast_2d(b), axis=1, kind="stable"), axis=1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    rho = (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return (1 - rho) / 2


def div(a, b, p):
    """p = 'sp' -> Spearman divergence, else RBO divergence with weight p."""
    return d_sp_batch(a, b) if p == "sp" else 1.0 - rbo_ext_batch(a, b, p)


def pair_idx(B, n_pairs, rng):
    i = rng.integers(0, B, n_pairs)
    return i, (i + rng.integers(1, B, n_pairs)) % B


class Expl:
    """SHAP matrix of one model on a fixed flow set + cached bootstrap importances."""

    def __init__(self, phi, labels, classes, B=1000, n_pairs=5000, seed=0):
        self.phi, self.y = phi, labels
        rng = np.random.default_rng(seed)
        self.boot = {c: bootstrap_importances(phi[labels == c], B, rng) for c in classes}
        self.pairs = pair_idx(B, n_pairs, rng)
        self._N = {}

    def s(self, c):
        return importance(self.phi[self.y == c])

    def N(self, c, p):
        if (c, p) not in self._N:
            i, j = self.pairs
            self._N[(c, p)] = float(div(self.boot[c][i], self.boot[c][j], p).mean())
        return self._N[(c, p)]


def eed(e1, e2, c, p):
    return float(div(e1.s(c), e2.s(c), p)[0]) - 0.5 * (e1.N(c, p) + e2.N(c, p))


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def ttest_p(v):
    v = np.asarray(v, float)
    return float(stats.ttest_1samp(v, 0.0).pvalue) if len(v) > 1 and v.std() > 0 else 1.0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="S1,S2,S3,S4,S5,S6,S7")
    parts = set(ap.parse_args().only.upper().split(","))
    cfg = yaml.safe_load(open("configs/protocol.yaml")); ex = cfg["explanation"]; od = cfg["data"]["out_dir"]
    B, NP_, MINC = ex["bootstrap_B"], ex["noise_pairs"], ex["min_class_samples"]
    seeds, K = cfg["seeds"], cfg["partition"]["n_clients"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, idx, feats = load_2017(od, dev)
    R, bg = idx["R"], idx["background"]; yR = y[R].cpu().numpy(); Xbg = X[bg]
    cls = [c for c in range(C) if (yR == c).sum() >= MINC]; CN = [CLASSES[c] for c in cls]
    os.makedirs(CACHE, exist_ok=True); os.makedirs("results/analysis", exist_ok=True)
    out_json = "results/analysis/sensitivity.json"
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    t0 = time.time(); timing = {}
    log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {m}", flush=True)
    run = lambda n: os.path.join("results", "runs", n, "final.pt")
    models = {f"cen{s}": f"central_s{s}" for s in seeds}
    models.update({f"f100_{s}": f"fedavg_dir_a100_s{s}" for s in seeds})
    models.update({f"f01_{s}": f"fedavg_dir_a0.1_s{s}" for s in seeds})
    _mcache = {}

    def model(key):
        if key not in _mcache:
            _mcache.clear(); _mcache[key] = load_model(run(models[key]), X.shape[1], dev)
        return _mcache[key]

    def phi_R(key, bg_name="main", bg_X=None):
        f = os.path.join(CACHE, f"R_{bg_name}_{key}.npy")
        if os.path.exists(f): return np.load(f)
        t = time.time()
        ph = expected_gradients(model(key), X[R], Xbg if bg_X is None else bg_X, y[R], nsamples=NS, seed=0)
        timing.setdefault("eg_on_R_s", []).append(time.time() - t)
        np.save(f, ph); return ph

    def save():
        json.dump(res, open(out_json, "w"), indent=1, default=float)

    def core(E, p, keys_c=("cen",), tag=""):
        """Seed baseline, federation excess (alpha=100) and drift (alpha=0.1) for one divergence."""
        base = {c: np.mean([eed(E[f"cen{a}"], E[f"cen{b}"], c, p) for a, b in itertools.combinations(seeds, 2)]) for c in cls}
        b100 = {c: np.mean([eed(E[f"f100_{a}"], E[f"f100_{b}"], c, p) for a, b in itertools.combinations(seeds, 2)]) for c in cls}
        fed = {c: [np.mean([eed(E[f"f100_{s}"], E[f"cen{s2}"], c, p) for s2 in seeds if s2 != s]) - base[c] for s in seeds] for c in cls}
        dr = {c: [np.mean([eed(E[f"f01_{s}"], E[f"f100_{s2}"], c, p) for s2 in seeds if s2 != s]) - b100[c] for s in seeds] for c in cls}
        fed_p, dr_p = holm([ttest_p(fed[c]) for c in cls]), holm([ttest_p(dr[c]) for c in cls])
        return {"seed_baseline": {CLASSES[c]: float(base[c]) for c in cls},
                "federation_excess": {CLASSES[c]: {"ci": t_ci(fed[c]), "holm_p": fed_p[i]} for i, c in enumerate(cls)},
                "drift_alpha0.1": {CLASSES[c]: {"ci": t_ci(dr[c]), "holm_p": dr_p[i]} for i, c in enumerate(cls)},
                "federation_excess_classavg": t_ci(np.mean([fed[c] for c in cls], axis=0)),
                "drift_alpha0.1_classavg": t_ci(np.mean([dr[c] for c in cls], axis=0))}

    # ---------- explanations on R with the main background (used by S1, S3, S4) ----------
    E = None
    if parts & {"S1", "S3", "S4", "S6"}:
        E = {}
        for k in models:
            E[k] = Expl(phi_R(k), yR, cls, B, NP_, seed=0)
        log(f"main-background explanations ready ({len(E)} models)")

    # ---------- S1: RBO p and Spearman ----------
    if "S1" in parts:
        S1 = {}
        for p in PS + ["sp"]:
            S1[str(p)] = core(E, p); log(f"S1 p={p} core done")
        # class-mix share (same global model, client flows) for alpha in {1, 0.1}
        mix = {str(p): {} for p in PS + ["sp"]}
        for a in ["1", "0.1"]:
            dR = {str(p): [] for p in PS + ["sp"]}; dM = {str(p): [] for p in PS + ["sp"]}
            for s in seeds:
                m = load_model(run(f"fedavg_dir_a{a}_s{s}"), X.shape[1], dev)
                f = os.path.join(CACHE, f"R_main_fa{a}_{s}.npy")
                if os.path.exists(f): phR = np.load(f)
                else:
                    phR = expected_gradients(m, X[R], Xbg, y[R], nsamples=NS, seed=0); np.save(f, phR)
                means = {c: np.abs(phR[yR == c]).mean(0) for c in range(C) if (yR == c).any()}
                sR = importance(phR)
                _, tb = load_partition(od, f"dir_a{a}_s{s}", dev, K)
                for ci, ib in enumerate(tb):
                    fc = os.path.join(CACHE, f"client_a{a}_s{s}_c{ci}.npz")
                    if os.path.exists(fc):
                        z = np.load(fc); phi, yk = z["phi"], z["y"]
                    else:
                        g = torch.Generator().manual_seed(1000 * s + ci)
                        sel = ib[torch.randperm(len(ib), generator=g)[:ex["max_explained_per_client"]].to(dev)]
                        t = time.time()
                        phi = expected_gradients(m, X[sel], Xbg, y[sel], nsamples=NS, seed=ci)
                        timing.setdefault("eg_client500_s", []).append(time.time() - t)
                        yk = y[sel].cpu().numpy(); np.savez(fc, phi=phi, y=yk)
                    pi = np.bincount(yk, minlength=C) / len(yk)
                    pred = sum(pi[c] * means[c] for c in means); pred = pred / pred.sum()
                    sk = importance(phi)
                    for p in PS + ["sp"]:
                        dR[str(p)].append(float(div(sk, sR, p)[0])); dM[str(p)].append(float(div(sk, pred, p)[0]))
            for p in PS + ["sp"]:
                mix[str(p)][a] = {"D_vs_R": float(np.mean(dR[str(p)])), "D_vs_mix": float(np.mean(dM[str(p)])),
                                  "share_explained_by_mix": 1 - float(np.mean(dM[str(p)]) / np.mean(dR[str(p)]))}
            log(f"S1 class-mix alpha={a} done")
        for p in PS + ["sp"]:
            S1[str(p)]["class_mix"] = mix[str(p)]
        # agreement of per-class conclusions with the main setting (p = 0.9)
        ref = [S1["0.9"]["federation_excess"][cn]["ci"][0] for cn in CN] + [S1["0.9"]["drift_alpha0.1"][cn]["ci"][0] for cn in CN]
        for p in PS + ["sp"]:
            v = [S1[str(p)]["federation_excess"][cn]["ci"][0] for cn in CN] + [S1[str(p)]["drift_alpha0.1"][cn]["ci"][0] for cn in CN]
            S1[str(p)]["spearman_with_p0.9_over_class_effects"] = float(stats.spearmanr(ref, v).statistic)
            S1[str(p)]["n_significant_fed_excess"] = int(sum(S1[str(p)]["federation_excess"][cn]["holm_p"] < .05 for cn in CN))
            S1[str(p)]["n_significant_drift"] = int(sum(S1[str(p)]["drift_alpha0.1"][cn]["holm_p"] < .05 for cn in CN))
        res["S1_rbo_p"] = S1; save()

    # ---------- S2: background size and draw ----------
    if "S2" in parts:
        Rn = R.cpu().numpy()

        def strat_bg(m, seed):
            rng = np.random.default_rng(seed); cl = np.unique(yR); q, r = divmod(m, len(cl)); out = []
            for i, c in enumerate(cl):
                out.append(rng.permutation(np.flatnonzero(yR == c))[: q + (1 if i < r else 0)])
            out = np.concatenate(out)
            if len(out) < m:
                out = np.concatenate([out, rng.permutation(np.setdiff1d(np.arange(len(yR)), out))[: m - len(out)]])
            return torch.from_numpy(Rn[np.sort(out)]).to(dev)

        S2 = {}
        configs = [("m25", 25, 0), ("m50", 50, 0), ("main", 100, None), ("m200", 200, 0), ("m100_drawB", 100, 1), ("m100_drawC", 100, 2)]
        Eb = {}
        for name, m, sd in configs:
            bX = Xbg if sd is None else X[strat_bg(m, sd)]
            Eb[name] = {k: Expl(phi_R(k, name, bX), yR, cls, B, NP_, seed=0) for k in models}
            if name in ("m25", "m50", "main", "m200"):
                S2[name] = core(Eb[name], 0.9)
            log(f"S2 background {name} done")
        same = {}
        for other in ["m100_drawB", "m100_drawC"]:
            same[other] = {CLASSES[c]: t_ci([eed(Eb["main"][k], Eb[other][k], c, 0.9) for k in models]) for c in cls}
        S2["same_model_two_shared_backgrounds_EED"] = same
        res["S2_background"] = S2; save()

    # ---------- S3: number of explained flows ----------
    if "S3" in parts:
        rng = np.random.default_rng(3); S3 = {"by_n": {}, "threshold_cells": {}}
        pairs = [(seeds[i], seeds[i + 1]) for i in range(0, len(seeds) - 1, 2)]
        for c in cls:
            nc = int((yR == c).sum()); S3["by_n"][CLASSES[c]] = {}
            for n in [10, 20, 30, 50, 100]:
                if n > nc: continue
                floors, eeds = [], []
                for (a, b) in pairs:
                    for _ in range(10):
                        sub = rng.choice(np.flatnonzero(yR == c), n, replace=False)
                        ea = Expl(E[f"cen{a}"].phi[sub], np.full(n, c), [c], B, NP_, seed=int(rng.integers(1e9)))
                        eb = Expl(E[f"cen{b}"].phi[sub], np.full(n, c), [c], B, NP_, seed=int(rng.integers(1e9)))
                        floors.append(ea.N(c, 0.9)); eeds.append(eed(ea, eb, c, 0.9))
                S3["by_n"][CLASSES[c]][n] = {"floor_mean": float(np.mean(floors)), "floor_sd": float(np.std(floors, ddof=1)),
                                             "eed_mean": float(np.mean(eeds)), "eed_sd": float(np.std(eeds, ddof=1))}
            full = [eed(E[f"cen{a}"], E[f"cen{b}"], c, 0.9) for a, b in pairs]
            S3["by_n"][CLASSES[c]]["all"] = {"n": nc, "floor_mean": float(np.mean([E[f"cen{a}"].N(c, 0.9) for a, _ in pairs])),
                                            "eed_mean": float(np.mean(full))}
            log(f"S3 class {CLASSES[c]} done")
        yn = y.cpu().numpy()
        for a in ["1", "0.1"]:
            S3["threshold_cells"][a] = {}
            for T in [10, 30, 50, 100]:
                cells, per_class = 0, {cn: 0 for cn in CLASSES}
                for s in seeds:
                    _, tb = load_partition(od, f"dir_a{a}_s{s}", "cpu", K)
                    for ib in tb:
                        cnt = np.bincount(yn[ib.numpy()], minlength=C)
                        for c in range(C):
                            if cnt[c] >= T: cells += 1; per_class[CLASSES[c]] += 1
                S3["threshold_cells"][a][T] = {"cells_total": cells, "per_class": per_class,
                                                "max_cells": len(seeds) * K * C}
        res["S3_n_explained"] = S3; save()

    # ---------- S4: bootstrap configuration ----------
    if "S4" in parts:
        S4 = {}
        phi0 = E[f"cen{seeds[0]}"].phi
        for c in cls:
            S4[CLASSES[c]] = {}
            A = phi0[yR == c]
            for Bv in [200, 500, 1000, 2000]:
                for npairs in [1000, 5000, 20000]:
                    fl = []
                    for r in range(10):
                        rng = np.random.default_rng(100 + r)
                        bt = bootstrap_importances(A, Bv, rng); i, j = pair_idx(Bv, npairs, rng)
                        fl.append(float(div(bt[i], bt[j], 0.9).mean()))
                    S4[CLASSES[c]][f"B{Bv}_pairs{npairs}"] = {"mean": float(np.mean(fl)), "sd": float(np.std(fl, ddof=1))}
            log(f"S4 class {CLASSES[c]} done")
        res["S4_bootstrap"] = S4; save()

    # ---------- S5: KernelSHAP per class ----------
    if "S5" in parts:
        try:
            import shap
            g = torch.Generator().manual_seed(0); NK = 20
            sel = torch.cat([R[torch.nonzero(y[R] == c).squeeze(1)[torch.randperm(int((yR == c).sum()), generator=g)[:NK].to(dev)]] for c in cls])
            xs, ys = X[sel], y[sel]; yk = ys.cpu().numpy(); bk = Xbg[:50].cpu().numpy()
            ks = seeds[:5]; keys = [f"cen{s}" for s in ks] + [f"f100_{s}" for s in ks] + [f"f01_{s}" for s in ks]
            EK, EE = {}, {}
            for k in keys:
                fk = os.path.join(CACHE, f"kernel_{k}.npy")
                if os.path.exists(fk): pk = np.load(fk)
                else:
                    m = model(k)
                    f = lambda a: m(torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dev)).detach().cpu().numpy()
                    sv = shap.KernelExplainer(f, bk).shap_values(xs.cpu().numpy(), nsamples=1000, silent=True)
                    sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)
                    pk = sv[np.arange(len(yk)), :, yk]; np.save(fk, pk)
                pe = expected_gradients(model(k), xs, Xbg, ys, nsamples=NS, seed=0)
                EK[k], EE[k] = Expl(pk, yk, cls, B, NP_), Expl(pe, yk, cls, B, NP_)
                log(f"S5 {k} done")
            S5 = {}
            for name, D in [("kernelshap", EK), ("expected_gradients_same_flows", EE)]:
                base = {c: np.mean([eed(D[f"cen{a}"], D[f"cen{b}"], c, 0.9) for a, b in itertools.combinations(ks, 2)]) for c in cls}
                b100 = {c: np.mean([eed(D[f"f100_{a}"], D[f"f100_{b}"], c, 0.9) for a, b in itertools.combinations(ks, 2)]) for c in cls}
                fed = {c: [np.mean([eed(D[f"f100_{s}"], D[f"cen{s2}"], c, 0.9) for s2 in ks if s2 != s]) - base[c] for s in ks] for c in cls}
                dr = {c: [np.mean([eed(D[f"f01_{s}"], D[f"f100_{s2}"], c, 0.9) for s2 in ks if s2 != s]) - b100[c] for s in ks] for c in cls}
                S5[name] = {"seed_baseline": {CLASSES[c]: float(base[c]) for c in cls},
                            "federation_excess": {CLASSES[c]: t_ci(fed[c]) for c in cls},
                            "drift_alpha0.1": {CLASSES[c]: t_ci(dr[c]) for c in cls},
                            "floor_mean": {CLASSES[c]: float(np.mean([D[k].N(c, 0.9) for k in keys])) for c in cls}}
            S5["rank_agreement_kernel_vs_eg"] = {CLASSES[c]: float(np.mean([stats.spearmanr(EK[k].s(c), EE[k].s(c)).statistic for k in keys])) for c in cls}
            S5["setup"] = {"flows_per_class": NK, "models": keys, "kernel_nsamples": 1000, "kernel_background": 50}
            res["S5_kernelshap"] = S5; save()
        except ImportError:
            res["S5_kernelshap"] = "skipped: pip install shap"; save()

    # ---------- S6: cost ----------
    if "S6" in parts:
        k0 = f"cen{seeds[0]}"; t = time.time()
        expected_gradients(model(k0), X[R], Xbg, y[R], nsamples=NS, seed=0)
        if dev == "cuda": torch.cuda.synchronize()
        t_eg = time.time() - t
        t = time.time(); Expl(E[k0].phi, yR, cls, B, NP_, seed=9).N(cls[0], 0.9); t_floor = time.time() - t
        F = X.shape[1]
        res["S6_cost"] = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                          "eg_seconds_R": t_eg, "R_flows": int(len(R)), "eg_nsamples": NS,
                          "floor_seconds_one_class": t_floor,
                          "summary_floats_per_class": F + 1, "summary_bytes_all_classes_float32": (F + 1) * len(cls) * 4,
                          "measured_runs": {k: float(np.mean(v)) for k, v in timing.items()}}
        save()

    # ---------- S7: per-class detection table ----------
    if "S7" in parts:
        def collect(name_fmt):
            fs = [os.path.join("results", "runs", name_fmt.format(s=s), "metrics.json") for s in seeds]
            fs = [f for f in fs if os.path.exists(f)]
            if not fs: return None
            T = [json.load(open(f))["testb"] for f in fs]
            pc = np.array([t["per_class_f1"] for t in T])
            return {"n_seeds": len(T), "macro_f1": t_ci([t["macro_f1"] for t in T]),
                    "per_class_f1": {CLASSES[c]: t_ci(pc[:, c]) for c in range(C)}, "support": T[0]["support"]}
        S7 = {"test_support": None, "R_counts": {CLASSES[c]: int((yR == c).sum()) for c in range(C)}}
        for lab, fmt in [("central", "central_s{s}"), ("fedavg_a100", "fedavg_dir_a100_s{s}"), ("fedavg_a0.1", "fedavg_dir_a0.1_s{s}"),
                         ("attack_skew", "fedavg_attack_skew_s{s}"), ("fedprox0.01_a0.1", "fedprox_mu0.01_dir_a0.1_s{s}")]:
            S7[lab] = collect(fmt)
            if S7[lab] and S7["test_support"] is None:
                S7["test_support"] = {CLASSES[c]: int(S7[lab]["support"][c]) for c in range(C)}
        res["S7_per_class"] = S7; save()

    # ---------- summary ----------
    L = []
    if "S1_rbo_p" in res:
        L.append("=== S1: weight p (class-avg federation excess a=100 | class-avg drift a=0.1 | mix share a=1 | #sig fed/drift | rho vs p=0.9) ===")
        for p, d in res["S1_rbo_p"].items():
            L.append(f"p={p:5s} fed {d['federation_excess_classavg'][0]:.3f}+-{d['federation_excess_classavg'][1]:.3f} | "
                     f"drift {d['drift_alpha0.1_classavg'][0]:.3f}+-{d['drift_alpha0.1_classavg'][1]:.3f} | "
                     f"mix {d['class_mix']['1']['share_explained_by_mix']:.2f}/{d['class_mix']['0.1']['share_explained_by_mix']:.2f} | "
                     f"{d['n_significant_fed_excess']}/{d['n_significant_drift']} of {len(CN)} | rho {d['spearman_with_p0.9_over_class_effects']:.2f}")
            L.append("        seed baseline: " + " ".join(f"{cn[:6]}:{v:.2f}" for cn, v in d["seed_baseline"].items()))
    if "S2_background" in res:
        L.append("\n=== S2: background size (class-avg federation excess | drift a=0.1) ===")
        for m, d in res["S2_background"].items():
            if m.startswith("same"): continue
            L.append(f"{m:5s} fed {d['federation_excess_classavg'][0]:.3f}+-{d['federation_excess_classavg'][1]:.3f} | drift "
                     f"{d['drift_alpha0.1_classavg'][0]:.3f}+-{d['drift_alpha0.1_classavg'][1]:.3f} | base " +
                     " ".join(f"{cn[:6]}:{v:.2f}" for cn, v in d["seed_baseline"].items()))
        for o, d in res["S2_background"]["same_model_two_shared_backgrounds_EED"].items():
            L.append(f"same model, main vs {o}: " + " ".join(f"{cn[:6]}:{v[0]:.3f}" for cn, v in d.items()))
    if "S3_n_explained" in res:
        L.append("\n=== S3: explained flows n -> floor (sd) | EED between two centralized seeds (sd) ===")
        for cn, d in res["S3_n_explained"]["by_n"].items():
            L.append(f"{cn:10s} " + " ".join(f"n={n}: {v['floor_mean']:.3f}({v.get('floor_sd', 0):.3f})|{v['eed_mean']:.3f}({v.get('eed_sd', 0):.3f})"
                                             for n, v in d.items() if n != "all") + f" | all n={d['all']['n']}: {d['all']['floor_mean']:.3f}|{d['all']['eed_mean']:.3f}")
        for a, d in res["S3_n_explained"]["threshold_cells"].items():
            L.append(f"a={a}: client-class cells >= T: " + " ".join(f"T={T}:{v['cells_total']}" for T, v in d.items()))
    if "S4_bootstrap" in res:
        L.append("\n=== S4: floor mean (MC sd over 10 repeats) ===")
        for cn, d in res["S4_bootstrap"].items():
            L.append(f"{cn:10s} " + " ".join(f"{k}:{v['mean']:.4f}({v['sd']:.4f})" for k, v in d.items()))
    if isinstance(res.get("S5_kernelshap"), dict):
        k5 = res["S5_kernelshap"]
        L.append("\n=== S5: KernelSHAP vs EG on the same 20 flows/class (seed baseline | federation excess | drift) ===")
        for name in ["kernelshap", "expected_gradients_same_flows"]:
            d = k5[name]
            L.append(f"{name}:")
            for cn in d["seed_baseline"]:
                L.append(f"   {cn:10s} base {d['seed_baseline'][cn]:.3f} | fed {d['federation_excess'][cn][0]:.3f}+-{d['federation_excess'][cn][1]:.3f}"
                         f" | drift {d['drift_alpha0.1'][cn][0]:.3f}+-{d['drift_alpha0.1'][cn][1]:.3f} | floor {d['floor_mean'][cn]:.3f}")
        L.append("rank agreement (Spearman) " + " ".join(f"{cn[:6]}:{v:.2f}" for cn, v in k5["rank_agreement_kernel_vs_eg"].items()))
    if "S6_cost" in res:
        c6 = res["S6_cost"]
        L.append(f"\n=== S6: {c6['device']}: EG on R ({c6['R_flows']} flows) {c6['eg_seconds_R']:.1f}s | floor/class {c6['floor_seconds_one_class']:.2f}s | "
                 f"summary {c6['summary_bytes_all_classes_float32']} bytes; {c6['measured_runs']}")
    if "S7_per_class" in res:
        s7 = res["S7_per_class"]
        L.append("\n=== S7: per-class Test-B support | R counts | F1 ===")
        L.append("           " + " ".join(f"{cn[:8]:>9s}" for cn in CLASSES))
        L.append("support    " + " ".join(f"{s7['test_support'][cn]:9d}" for cn in CLASSES))
        L.append("R          " + " ".join(f"{s7['R_counts'][cn]:9d}" for cn in CLASSES))
        for lab in ["central", "fedavg_a100", "fedavg_a0.1", "attack_skew", "fedprox0.01_a0.1"]:
            if s7.get(lab):
                L.append(f"{lab[:10]:10s} " + " ".join(f"{s7[lab]['per_class_f1'][cn][0]:9.3f}" for cn in CLASSES) + f"  (n={s7[lab]['n_seeds']})")
    open("results/analysis/sensitivity_summary.txt", "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L))
    log("all done")


if __name__ == "__main__":
    main()
