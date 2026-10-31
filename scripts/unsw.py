"""Camera-ready review item (R5, R6, R8): replicate the protocol on an IDS dataset from a
different source and feature space: UNSW-NB15 (UNSW Canberra, IXIA PerfectStorm traffic,
Argus/Bro features; 10 classes), using the SAME protocol as CICIDS2017: dedupe before a
stratified 70/10/20 split, reference set R (<=200 per class) and a shared 100-flow
background from the 10% split, federated normalisation, 10 clients, Dirichlet alpha in
{100, 5, 1, 0.5, 0.3, 0.1}, 50 rounds, MLP (d -> 128 -> 64 -> C), expected gradients
(1,000 samples), RBO p = 0.9, bootstrap noise floor, minimum 30 explained flows.

Data: put the two official CSVs (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv;
175,341 + 82,332 rows, with an 'attack_cat' column) in data/raw/unswnb15/ .

Usage (from the fedxai-ids folder):
  python scripts/unsw.py prepare      # ~1 min: clean, split, R, partitions
  python scripts/unsw.py train        # ~2 h on RTX 4050, resumable (finished runs skipped)
  python scripts/unsw.py analyze      # ~30-60 min
  python scripts/unsw.py all          # all three in order
  add --seeds 5 to use seeds 0-4 only (faster)
Output: data/processed/unsw/, results/unsw/runs/, results/unsw/unsw_analysis.json + unsw_summary.txt
"""
import argparse, glob, itertools, json, os, sys, time
import numpy as np, pandas as pd, torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.data.prepare import stratified_split
from fedxai.data.fed_norm import NormStats, federated_norm_stats, build_reference
from fedxai.data.partition import dirichlet_partition, local_test_split
from fedxai.train.data import class_weights
from fedxai.train.model import MLP
from fedxai.train.fl import train_centralized, train_federated
from fedxai.train.evaluate import detection_metrics
from fedxai.explain.eg import expected_gradients
from fedxai.metrics.agreement import importance, d_rbo, bootstrap_importances, within_divergences, client_eed, t_ci

RAW, OUT, RES = "data/raw/unswnb15", "data/processed/unsw", "results/unsw"
ALPHAS = [100, 5, 1, 0.5, 0.3, 0.1]
PROX_ALPHAS, MU = [100, 0.1], 0.01
K, ROUNDS, P, B, NPAIRS, MINC = 10, 50, 0.9, 1000, 5000, 30
NS = int(os.environ.get("FEDXAI_NS", 1000))      # expected-gradients samples (env override only for smoke tests)
ROUNDS = int(os.environ.get("FEDXAI_ROUNDS", ROUNDS))
B, NPAIRS = int(os.environ.get("FEDXAI_B", B)), int(os.environ.get("FEDXAI_PAIRS", NPAIRS))
PER_CLASS_REF, BG_SIZE, MAX_EXPL, PER_CLASS_CLIENT, MIN_CLIENT = 200, 100, 500, 200, 500
CAT_COLS, TOP_CATS = ["proto", "service", "state"], 8
DROP_COLS = {"id", "label", "attack_cat", "srcip", "sport", "dstip", "dsport", "stime", "ltime"}
LABEL_MAP = {"normal": "Normal", "": "Normal", "nan": "Normal", "generic": "Generic", "exploits": "Exploits",
             "fuzzers": "Fuzzers", "dos": "DoS", "reconnaissance": "Reconnaissance", "analysis": "Analysis",
             "backdoor": "Backdoor", "backdoors": "Backdoor", "shellcode": "Shellcode", "worms": "Worms"}


# ============================================================ prepare
def prepare(seeds):
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    if not files: raise FileNotFoundError(f"put UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in {RAW}")
    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "attack_cat" not in df.columns:
            raise ValueError(f"{f}: no 'attack_cat' column. Use the training-set / testing-set CSVs.")
        dfs.append(df); print(f"read {os.path.basename(f)}: {len(df):,} rows")
    df = pd.concat(dfs, ignore_index=True, join="inner")
    rec = {"files": [os.path.basename(f) for f in files], "rows_raw": int(len(df))}
    lab = df["attack_cat"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    if lab.isna().any(): raise ValueError(f"unknown labels: {sorted(df['attack_cat'][lab.isna()].astype(str).unique())}")
    feats_raw = [c for c in df.columns if c not in DROP_COLS]
    X = df[feats_raw].copy()
    for c in feats_raw:
        X[c] = X[c].astype(str).str.strip() if c in CAT_COLS else pd.to_numeric(X[c], errors="coerce")
    num = [c for c in feats_raw if c not in CAT_COLS]
    X[num] = X[num].replace([np.inf, -np.inf], np.nan)
    bad = X[num].isna().any(axis=1); rec["rows_removed_inf_nan"] = int(bad.sum())
    X, lab = X.loc[~bad], lab.loc[~bad]
    Xy = X.assign(_y=lab.values)
    dup = Xy.duplicated(keep="first"); rec["rows_removed_duplicates"] = int(dup.sum()); Xy = Xy.loc[~dup]
    conf = Xy.drop(columns="_y").duplicated(keep=False); rec["rows_removed_label_conflicts"] = int(conf.sum()); Xy = Xy.loc[~conf]
    y = Xy.pop("_y").reset_index(drop=True); X = Xy.reset_index(drop=True)
    rec["rows_final"] = int(len(X)); rec["class_counts_final"] = y.value_counts().to_dict()
    tr, ta, tb = stratified_split(y, (0.70, 0.10, 0.20), 0)
    # one-hot of categorical columns; categories decided on TRAIN only (top 8 + 'other')
    for c in CAT_COLS:
        if c not in X.columns: continue
        top = X[c].iloc[tr].value_counts().index[:TOP_CATS].tolist()
        v = X[c].where(X[c].isin(top), "other")
        for t in top + ["other"]:
            X[f"{c}={t}"] = (v == t).astype(np.float64)
        X = X.drop(columns=c)
    X = X.astype(np.float64)
    const = [c for c in X.columns if X[c].iloc[tr].nunique() <= 1]; X = X.drop(columns=const)
    rec["columns_removed_constant_train"] = const
    feats = list(X.columns)
    classes = ["Normal"] + [c for c in y.iloc[tr].value_counts().index if c != "Normal"]
    yv = y.to_numpy()
    R_in, bg_in = build_reference(yv[ta], PER_CLASS_REF, BG_SIZE, 0)
    R, bg = ta[R_in], ta[bg_in]
    rec.update({"n_features": len(feats), "classes": classes,
                "split_sizes": {"train": len(tr), "test_a": len(ta), "test_b": len(tb)},
                "R_class_counts": pd.Series(yv[R]).value_counts().to_dict()})
    os.makedirs(os.path.join(OUT, "partitions"), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "unsw.npz"), X=X.to_numpy(), y=yv.astype(str), train=tr, test_a=ta,
                        test_b=tb, R=R, background=bg, features=np.array(feats), classes=np.array(classes))
    ns = federated_norm_stats([X.to_numpy()[tr]], True)
    json.dump({"features": feats, **ns.to_dict()}, open(os.path.join(OUT, "norm_stats.json"), "w"))
    ytr, ytb = yv[tr], yv[tb]; cl = list(np.unique(ytr))
    for s in seeds:
        for a in ALPHAS:
            parts, Pm = dirichlet_partition(ytr, K, a, s, MIN_CLIENT)
            tparts = local_test_split(ytb, Pm, cl, s)
            np.savez_compressed(os.path.join(OUT, "partitions", f"dir_a{a}_s{s}.npz"),
                                **{f"train_{k}": tr[p] for k, p in enumerate(parts)},
                                **{f"testb_{k}": tb[p] for k, p in enumerate(tparts)})
    json.dump(rec, open(os.path.join(OUT, "lock_record_unsw.json"), "w"), indent=1, default=str)
    print(json.dumps({k: rec[k] for k in ["rows_raw", "rows_removed_duplicates", "rows_final", "n_features",
                                          "class_counts_final", "R_class_counts"]}, indent=1, default=str))


def load(dev):
    z = np.load(os.path.join(OUT, "unsw.npz"), allow_pickle=False)
    ns = json.load(open(os.path.join(OUT, "norm_stats.json")))
    st = NormStats(np.array(ns["log1p_mask"]), np.array(ns["mean"]), np.array(ns["std"]))
    classes = list(z["classes"]); ci = {c: i for i, c in enumerate(classes)}
    X = torch.from_numpy(st.transform(z["X"])).to(dev)
    y = torch.from_numpy(np.array([ci[v] for v in z["y"]], dtype=np.int64)).to(dev)
    idx = {k: torch.from_numpy(z[k].astype(np.int64)).to(dev) for k in ["train", "test_a", "test_b", "R", "background"]}
    return X, y, idx, list(z["features"]), classes


def load_part(name, dev):
    z = np.load(os.path.join(OUT, "partitions", name + ".npz"))
    t = lambda k: torch.from_numpy(z[k].astype(np.int64)).to(dev)
    return [t(f"train_{k}") for k in range(K)], [t(f"testb_{k}") for k in range(K)]


# ============================================================ train
def train(seeds):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, idx, feats, classes = load(dev); C = len(classes)
    w = class_weights(y[idx["train"]], C)
    runs = []
    for s in seeds:
        runs.append(("central", None, s, 0.0))
        runs += [("fedavg", a, s, 0.0) for a in ALPHAS]
        runs += [("fedprox", a, s, MU) for a in PROX_ALPHAS]
    name = lambda k, a, s, mu: f"central_s{s}" if k == "central" else (f"fedavg_dir_a{a}_s{s}" if k == "fedavg" else f"fedprox_mu{mu}_dir_a{a}_s{s}")
    todo = [r for r in runs if not os.path.exists(os.path.join(RES, "runs", name(*r), "metrics.json"))]
    print(f"{len(runs)} runs, {len(todo)} to do"); t_all = time.time()
    for i, (k, a, s, mu) in enumerate(todo, 1):
        n = name(k, a, s, mu); rd = os.path.join(RES, "runs", n); os.makedirs(rd, exist_ok=True)
        torch.manual_seed(s); np.random.seed(s)
        m = MLP(X.shape[1], C).to(dev); t = time.time(); meta = {"run": n, "seed": s, "mu": mu}
        if k == "central":
            train_centralized(m, X, y, idx["train"], ROUNDS, w, seed=s, log=lambda *_: None)
        else:
            tr, tb = load_part(f"dir_a{a}_s{s}", dev)
            train_federated(m, X, y, tr, ROUNDS, w, mu=mu, seed=s, log=lambda *_: None)
            meta["per_client_testb"] = [detection_metrics(m, X, y, ib, C) for ib in tb]
        meta["train_time_s"] = round(time.time() - t, 1)
        meta["testb"] = detection_metrics(m, X, y, idx["test_b"], C)
        torch.save(m.state_dict(), os.path.join(rd, "final.pt"))
        json.dump(meta, open(os.path.join(rd, "metrics.json"), "w"), indent=1)
        el = time.time() - t_all
        print(f"[{i}/{len(todo)}] {n}: macro-F1 {meta['testb']['macro_f1']:.4f} | {meta['train_time_s']:.0f}s | "
              f"ETA {(len(todo) - i) * el / i / 3600:.2f} h", flush=True)


# ============================================================ analyze
def holm(pv):
    p = np.asarray(pv, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def tp(v):
    v = np.asarray(v, float)
    return float(stats.ttest_1samp(v, 0).pvalue) if len(v) > 1 and v.std() > 0 else 1.0


def analyze(seeds):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, idx, feats, classes = load(dev); C = len(classes)
    R, bgi = idx["R"], idx["background"]; yR = y[R].cpu().numpy(); Xbg = X[bgi]
    cls = [c for c in range(C) if (yR == c).sum() >= MINC]; CN = [classes[c] for c in cls]
    rng = np.random.default_rng(0); t0 = time.time()
    log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {m}", flush=True)
    rp = lambda n: os.path.join(RES, "runs", n)
    cache = os.path.join(RES, "cache"); os.makedirs(cache, exist_ok=True)

    def mdl(n):
        m = MLP(X.shape[1], C).to(dev)
        m.load_state_dict(torch.load(os.path.join(rp(n), "final.pt"), map_location=dev, weights_only=True)); m.eval(); return m

    phiR, W = {}, {}

    def prep(n):
        if n in phiR: return
        f = os.path.join(cache, f"R_{n}.npy")
        if os.path.exists(f): phiR[n] = np.load(f)
        else: phiR[n] = expected_gradients(mdl(n), X[R], Xbg, y[R], nsamples=NS, seed=0); np.save(f, phiR[n])
        for c in cls:
            W[(n, c)] = within_divergences(bootstrap_importances(phiR[n][yR == c], B, rng), NPAIRS, rng, P)

    def pe(a, b, c):
        mk = yR == c
        return d_rbo(importance(phiR[a][mk]), importance(phiR[b][mk]), P) - 0.5 * (W[(a, c)].mean() + W[(b, c)].mean())

    cen = [f"central_s{s}" for s in seeds]; fa = lambda a, s: f"fedavg_dir_a{a}_s{s}"; fp = lambda a, s: f"fedprox_mu{MU}_dir_a{a}_s{s}"
    for n in cen: prep(n)
    for a in ALPHAS:
        for s in seeds: prep(fa(a, s))
    for a in PROX_ALPHAS:
        for s in seeds: prep(fp(a, s))
    log("explanations on R done")
    res = {"classes": classes, "classes_analyzed": CN, "n_features": len(feats), "seeds": seeds,
           "R_counts": {classes[c]: int((yR == c).sum()) for c in range(C)}}

    # A. seed baseline
    base = {c: [pe(a, b, c) for a, b in itertools.combinations(cen, 2)] for c in cls}
    res["A_seed_baseline"] = {classes[c]: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for c, v in base.items()}
    bm = {c: float(np.mean(v)) for c, v in base.items()}

    # B. fidelity excess per alpha (federation itself = alpha 100)
    fid = {}
    for a in ALPHAS:
        per = {c: [np.mean([pe(fa(a, s), f"central_s{s2}", c) for s2 in seeds if s2 != s]) - bm[c] for s in seeds] for c in cls}
        ph = holm([tp(per[c]) for c in cls])
        fid[str(a)] = {"per_class": {classes[c]: {"ci": t_ci(per[c]), "holm_p": ph[i]} for i, c in enumerate(cls)},
                       "class_avg": t_ci(np.mean([per[c] for c in cls], axis=0))}
    res["B_fidelity_excess"] = fid

    # C. detection
    det = {}

    def det_of(names):
        T = [json.load(open(os.path.join(rp(n), "metrics.json")))["testb"] for n in names]
        pc = np.array([t["per_class_f1"] for t in T])
        return {"macro_f1": t_ci([t["macro_f1"] for t in T]), "per_class_f1": {classes[c]: t_ci(pc[:, c]) for c in range(C)},
                "support": T[0]["support"], "values": [t["macro_f1"] for t in T]}
    det["central"] = det_of(cen)
    for a in ALPHAS: det[f"fedavg_a{a}"] = det_of([fa(a, s) for s in seeds])
    for a in PROX_ALPHAS: det[f"fedprox_a{a}"] = det_of([fp(a, s) for s in seeds])
    res["C_detection"] = {k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in det.items()}
    log("A-C done")

    # F. heterogeneity drift vs alpha = 100 (minus the alpha=100 cross-seed baseline)
    f100 = [fa(100, s) for s in seeds]
    b100 = {c: float(np.mean([pe(a, b, c) for a, b in itertools.combinations(f100, 2)])) for c in cls}
    drift, avg_p, run_drift = {}, [], {}
    for a in ALPHAS[1:]:
        per = {c: [np.mean([pe(fa(a, s), fa(100, s2), c) for s2 in seeds if s2 != s]) - b100[c] for s in seeds] for c in cls}
        avg = np.mean([per[c] for c in cls], axis=0); run_drift[str(a)] = avg.tolist()
        drift[str(a)] = {"class_avg": t_ci(avg), "per_class": {classes[c]: t_ci(per[c]) for c in cls},
                         "per_class_p": {classes[c]: tp(per[c]) for c in cls}}
        avg_p.append(tp(avg))
    for a, hp in zip(ALPHAS[1:], holm(avg_p)): drift[str(a)]["class_avg_holm_p"] = hp
    allp = [(a, cn, drift[str(a)]["per_class_p"][cn]) for a in ALPHAS[1:] for cn in CN]
    for (a, cn, _), hp in zip(allp, holm([x[2] for x in allp])): drift[str(a)].setdefault("per_class_holm_p", {})[cn] = hp
    # detection drop vs alpha = 100, paired by seed
    f1_100 = np.array(det["fedavg_a100"]["values"])
    for a in ALPHAS[1:]:
        dd = f1_100 - np.array(det[f"fedavg_a{a}"]["values"]); drift[str(a)]["f1_drop"] = t_ci(dd); drift[str(a)]["f1_drop_p"] = tp(dd)
    for a, hp in zip(ALPHAS[1:], holm([drift[str(a)]["f1_drop_p"] for a in ALPHAS[1:]])): drift[str(a)]["f1_drop_holm_p"] = hp
    # run-level coupling (as in the main paper): raw and with partition means removed
    xs, ys_, xr, yr = [], [], [], []
    for a in ALPHAS[1:]:
        dvec = np.array(run_drift[str(a)]); fvec = f1_100.mean() - np.array(det[f"fedavg_a{a}"]["values"])
        xs += dvec.tolist(); ys_ += fvec.tolist(); xr += (dvec - dvec.mean()).tolist(); yr += (fvec - fvec.mean()).tolist()
    drift["coupling"] = {"spearman_raw": float(stats.spearmanr(xs, ys_).statistic),
                         "spearman_within_alpha": float(stats.spearmanr(xr, yr).statistic), "n_runs": len(xs)}
    res["F_drift"] = drift; log("F done")

    # H. FedProx minus FedAvg, paired by seed
    H = {}; cells = []
    for a in PROX_ALPHAS:
        df1 = np.array(det[f"fedprox_a{a}"]["values"]) - np.array(det[f"fedavg_a{a}"]["values"])
        H[str(a)] = {"delta_macro_f1": t_ci(df1), "p_f1": tp(df1), "per_class": {}}
        for c in cls:
            d = [np.mean([pe(fp(a, s), f"central_s{s2}", c) for s2 in seeds if s2 != s]) -
                 np.mean([pe(fa(a, s), f"central_s{s2}", c) for s2 in seeds if s2 != s]) for s in seeds]
            H[str(a)]["per_class"][classes[c]] = {"ci": t_ci(d), "p": tp(d)}; cells.append((a, classes[c]))
    for (a, cn), hp in zip(cells, holm([H[str(a)]["per_class"][cn]["p"] for a, cn in cells])): H[str(a)]["per_class"][cn]["holm_p"] = hp
    for a, hp in zip(PROX_ALPHAS, holm([H[str(a)]["p_f1"] for a in PROX_ALPHAS])): H[str(a)]["f1_holm_p"] = hp
    res["H_fedprox"] = H; log("H done")

    # D. class-mix effect + per-class client EED (same global model); BG. local background at alpha = 1
    mix, bgab = {}, {}
    for a in ALPHAS:
        dR, dM, pce = [], [], {cn: [] for cn in CN}
        same, cvr = {cn: [] for cn in CN}, {cn: [] for cn in CN}
        for s in seeds:
            n = fa(a, s); m = mdl(n); tr, tb = load_part(f"dir_a{a}_s{s}", dev)
            means = {c: np.abs(phiR[n][yR == c]).mean(0) for c in range(C) if (yR == c).any()}
            sR = importance(phiR[n])
            for ci, ib in enumerate(tb):
                g = torch.Generator().manual_seed(1000 * s + ci)
                sel = ib[torch.randperm(len(ib), generator=g)[:MAX_EXPL].to(dev)]
                phi = expected_gradients(m, X[sel], Xbg, y[sel], nsamples=NS, seed=ci)
                yk = y[sel].cpu().numpy(); pi = np.bincount(yk, minlength=C) / len(yk)
                pred = sum(pi[c] * means[c] for c in means)
                sk = importance(phi); dR.append(d_rbo(sk, sR, P)); dM.append(d_rbo(sk, pred / pred.sum(), P))
                yb = y[ib]
                lbg = X[tr[ci][torch.randperm(len(tr[ci]), generator=g)[:BG_SIZE].to(dev)]] if a == 1 else None
                for c in cls:
                    ic = ib[yb == c]
                    if len(ic) < MINC: continue
                    ic = ic[torch.randperm(len(ic), generator=g)[:PER_CLASS_CLIENT].to(dev)]
                    phc = expected_gradients(m, X[ic], Xbg, y[ic], nsamples=NS, seed=ci)
                    r = client_eed(phc, phiR[n][yR == c], p=P, B=B, n_pairs=NPAIRS, rng=rng, ref_within=W[(n, c)])
                    pce[classes[c]].append((r["EED"], r["significant"]))
                    if a == 1:
                        plo = expected_gradients(m, X[ic], lbg, y[ic], nsamples=NS, seed=ci)
                        same[classes[c]].append(client_eed(plo, phc, p=P, B=B, n_pairs=NPAIRS, rng=rng)["EED"])
                        cvr[classes[c]].append(client_eed(plo, phiR[n][yR == c], p=P, B=B, n_pairs=NPAIRS, rng=rng, ref_within=W[(n, c)])["EED"])
        mix[str(a)] = {"D_client_vs_R": t_ci(dR), "D_client_vs_mixture": t_ci(dM),
                       "share_explained_by_mix": 1 - float(np.mean(dM) / np.mean(dR)),
                       "per_class_client_EED": {cn: {"mean": float(np.mean([e for e, _ in v])) if v else None,
                                                     "sig_rate": float(np.mean([g_ for _, g_ in v])) if v else None,
                                                     "n_cells": len(v)} for cn, v in pce.items()}}
        if a == 1:
            bgab = {"local_vs_shared_same_flows": {cn: (float(np.mean(v)), len(v)) for cn, v in same.items() if v},
                    "client_local_bg_vs_R_shared": {cn: (float(np.mean(v)), len(v)) for cn, v in cvr.items() if v}}
        log(f"D alpha={a} done")
    res["D_mixture"], res["BG_background_alpha1"] = mix, bgab
    json.dump(res, open(os.path.join(RES, "unsw_analysis.json"), "w"), indent=1, default=float)

    # ---------------- summary ----------------
    L = [f"UNSW-NB15: {len(feats)} features, classes analysed (R >= {MINC}): {CN}", f"R counts: {res['R_counts']}",
         "\n=== Detection: macro-F1 (mean +- 95% t-CI) ==="]
    for k, v in det.items(): L.append(f"{k:16s} {v['macro_f1'][0]:.4f} +- {v['macro_f1'][1]:.4f}")
    L.append("per-class F1 (central | fedavg a100 | fedavg a0.1), test support:")
    for c in range(C):
        L.append(f"  {classes[c]:15s} {det['central']['per_class_f1'][classes[c]][0]:.3f} | {det['fedavg_a100']['per_class_f1'][classes[c]][0]:.3f} | "
                 f"{det['fedavg_a0.1']['per_class_f1'][classes[c]][0]:.3f}   n={det['central']['support'][c]}")
    L.append("\n=== A. Seed baseline (central vs central, per-class EED) ===")
    L.append(" ".join(f"{cn}:{v['mean']:.3f}" for cn, v in res["A_seed_baseline"].items()))
    L.append("\n=== B. Fidelity excess over seed baseline (class-avg; per class at alpha=100) ===")
    for a in ALPHAS: L.append(f"a={a:<5} {fid[str(a)]['class_avg'][0]:.3f} +- {fid[str(a)]['class_avg'][1]:.3f}")
    L.append("a=100 per class: " + " ".join(f"{cn}:{v['ci'][0]:.3f}{'*' if v['holm_p'] < .05 else ''}" for cn, v in fid["100"]["per_class"].items()))
    L.append("\n=== F. Drift vs alpha=100 (class-avg, Holm p) | F1 drop (Holm p) ===")
    for a in ALPHAS[1:]:
        d = drift[str(a)]
        L.append(f"a={a:<5} drift {d['class_avg'][0]:.3f} +- {d['class_avg'][1]:.3f} (p={d['class_avg_holm_p']:.3g}) | "
                 f"F1 drop {d['f1_drop'][0]:.3f} +- {d['f1_drop'][1]:.3f} (p={d['f1_drop_holm_p']:.3g}) | "
                 f"sig classes {sum(v < .05 for v in d['per_class_holm_p'].values())}/{len(CN)}")
    L.append(f"run-level Spearman drift vs F1 drop: raw {drift['coupling']['spearman_raw']:.2f}, "
             f"within alpha {drift['coupling']['spearman_within_alpha']:.2f} (n={drift['coupling']['n_runs']})")
    L.append("\n=== H. FedProx - FedAvg (dF1; per-class d fidelity, * Holm p<.05) ===")
    for a in PROX_ALPHAS:
        h = H[str(a)]
        L.append(f"a={a:<5} dF1 {h['delta_macro_f1'][0]:+.3f} (p={h['f1_holm_p']:.3g}) | " +
                 " ".join(f"{cn}:{v['ci'][0]:+.2f}{'*' if v['holm_p'] < .05 else ''}" for cn, v in h["per_class"].items()))
    L.append("\n=== D. Class mix (same global model): D vs R | D vs mixture | share ===")
    for a in ALPHAS:
        d = mix[str(a)]
        L.append(f"a={a:<5} {d['D_client_vs_R'][0]:.3f} | {d['D_client_vs_mixture'][0]:.3f} | {d['share_explained_by_mix']:.2f} | per-class client EED " +
                 " ".join(f"{cn[:6]}:{v['mean']:.3f}" for cn, v in d["per_class_client_EED"].items() if v["mean"] is not None))
    L.append("\n=== BG. alpha=1, client-local background vs shared (same flows) | client(local bg) vs R ===")
    L.append(" ".join(f"{cn[:6]}:{v[0]:.3f}" for cn, v in bgab["local_vs_shared_same_flows"].items()))
    L.append(" ".join(f"{cn[:6]}:{v[0]:.3f}" for cn, v in bgab["client_local_bg_vs_R_shared"].items()))
    open(os.path.join(RES, "unsw_summary.txt"), "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L)); log("all done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train", "analyze", "all"])
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); seeds = list(range(a.seeds))
    os.makedirs(RES, exist_ok=True)
    if a.stage in ("prepare", "all"): prepare(seeds)
    if a.stage in ("train", "all"): train(seeds)
    if a.stage in ("analyze", "all"): analyze(seeds)
