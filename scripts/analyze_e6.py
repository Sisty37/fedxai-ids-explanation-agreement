"""E6: cross-dataset transfer CICIDS2017 -> CSE-CIC-IDS2018 (no retraining).
2018 data are normalised with the 2017 statistics. For each model and each attack
family f present in both datasets (>= 30 samples in the 2018 sample and in R):
  dF1_f = F1_f on 2017 Test-B - F1_f on the 2018 sample
  dE_f  = EED between the model's explanations of class f on 2017 Test-B and on 2018
          (200 samples each, true-class logit, 2017 SHAP background, noise-floor corrected)
Association: Spearman over (family x seed) cells per model type, bootstrap 95% CI.
Reported as association, not prediction.
Usage: python scripts/analyze_e6.py     Output: results/analysis/e6.json + e6_summary.txt
"""
import json, os, sys, time
import numpy as np, torch, yaml
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, CLASSES
from fedxai.data.fed_norm import NormStats
from fedxai.train.model import MLP
from fedxai.train.evaluate import detection_metrics
from fedxai.explain.eg import expected_gradients
from fedxai.metrics.agreement import client_eed, t_ci

NS, PER = 1000, 200
C = len(CLASSES)
MODELS = {"central": "central_s{s}", "fedavg_a100": "fedavg_dir_a100_s{s}", "fedavg_a1": "fedavg_dir_a1_s{s}",
          "fedavg_a0.1": "fedavg_dir_a0.1_s{s}", "fedprox_a100": "fedprox_mu0.01_dir_a100_s{s}",
          "fedprox_a0.1": "fedprox_mu0.01_dir_a0.1_s{s}"}


def spearman_ci(x, y, n_boot=2000, seed=0):
    x, y = np.asarray(x), np.asarray(y); rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1: bs.append(stats.spearmanr(x[i], y[i]).statistic)
    return {"rho": float(stats.spearmanr(x, y).statistic), "ci": np.percentile(bs, [2.5, 97.5]).tolist(), "n": int(len(x))}


def main():
    cfg = yaml.safe_load(open("configs/protocol.yaml")); ex = cfg["explanation"]; out_dir = cfg["data"]["out_dir"]
    seeds = cfg["seeds"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, idx, feats = load_2017(out_dir, dev)
    z = np.load(os.path.join(out_dir, "cicids2018.npz"), allow_pickle=False)
    assert list(z["features"]) == list(feats), "2018 features must match 2017 order"
    ns = json.load(open(os.path.join(out_dir, "norm_stats_full.json")))
    st = NormStats(np.array(ns["log1p_mask"]), np.array(ns["mean"]), np.array(ns["std"]))
    X8 = torch.from_numpy(st.transform(z["X"])).to(dev)
    ci = {c: i for i, c in enumerate(CLASSES)}
    y8 = torch.from_numpy(np.array([ci[v] for v in z["y"]], dtype=np.int64)).to(dev)
    yR = y[idx["R"]].cpu().numpy(); Xbg = X[idx["background"]]
    tb = idx["test_b"]; ytb = y[tb]
    fams = [c for c in range(1, C) if (y8 == c).sum() >= ex["min_class_samples"] and (yR == c).sum() >= ex["min_class_samples"]]
    print("families analysed:", [CLASSES[c] for c in fams])
    g = torch.Generator().manual_seed(0)
    pick = lambda pool: pool[torch.randperm(len(pool), generator=g)[:PER].to(dev)]
    s17 = {c: pick(tb[ytb == c]) for c in fams}
    s18 = {c: pick(torch.nonzero(y8 == c).squeeze(1)) for c in fams}
    rng = np.random.default_rng(0); t0 = time.time(); res = {"families": [CLASSES[c] for c in fams], "models": {}}
    for mt, pat in MODELS.items():
        cells, det = [], []
        for s in seeds:
            rd = os.path.join("results", "runs", pat.format(s=s))
            if not os.path.exists(os.path.join(rd, "final.pt")): print("missing", rd); continue
            m = MLP(X.shape[1], C).to(dev)
            m.load_state_dict(torch.load(os.path.join(rd, "final.pt"), map_location=dev, weights_only=True)); m.eval()
            f17 = json.load(open(os.path.join(rd, "metrics.json")))["testb"]["per_class_f1"]
            d18 = detection_metrics(m, X8, y8, torch.arange(len(y8), device=dev), C)
            det.append({"seed": s, "macro_f1_2018_present_classes": float(np.mean([d18["per_class_f1"][c] for c in fams + [0]])),
                        "per_class_f1_2018": d18["per_class_f1"]})
            for c in fams:
                p17 = expected_gradients(m, X[s17[c]], Xbg, y[s17[c]], nsamples=NS, seed=1)
                p18 = expected_gradients(m, X8[s18[c]], Xbg, y8[s18[c]], nsamples=NS, seed=1)
                r = client_eed(p18, p17, p=ex["rbo_p"], B=ex["bootstrap_B"], n_pairs=ex["noise_pairs"], rng=rng)
                cells.append({"seed": s, "family": CLASSES[c], "dF1": float(f17[c] - d18["per_class_f1"][c]),
                              "dE": float(r["EED"]), "significant": r["significant"]})
        fam_sum = {CLASSES[c]: {"dF1": t_ci([x["dF1"] for x in cells if x["family"] == CLASSES[c]]),
                                "dE": t_ci([x["dE"] for x in cells if x["family"] == CLASSES[c]])} for c in fams}
        res["models"][mt] = {"cells": cells, "detection_2018": det, "per_family": fam_sum,
                             "association": spearman_ci([x["dF1"] for x in cells], [x["dE"] for x in cells])}
        print(f"[{(time.time() - t0) / 60:4.1f} min] {mt} done", flush=True)
    allc = [x for v in res["models"].values() for x in v["cells"]]
    res["association_all_models"] = spearman_ci([x["dF1"] for x in allc], [x["dE"] for x in allc])
    json.dump(res, open("results/analysis/e6.json", "w"), indent=1, default=float)
    L = ["=== E6: 2017 -> 2018 per family: dF1 (F1 drop) | dE (explanation shift, EED) ==="]
    for mt, v in res["models"].items():
        L.append(f"\n{mt}  association rho={v['association']['rho']:+.3f} "
                 f"[{v['association']['ci'][0]:+.3f}, {v['association']['ci'][1]:+.3f}] n={v['association']['n']}")
        for fn, d in v["per_family"].items():
            L.append(f"  {fn:11s} dF1 {d['dF1'][0]:+.3f}+-{d['dF1'][1]:.3f} | dE {d['dE'][0]:+.3f}+-{d['dE'][1]:.3f}")
    a = res["association_all_models"]
    L.append(f"\nall models pooled: rho={a['rho']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] n={a['n']}")
    open("results/analysis/e6_summary.txt", "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
