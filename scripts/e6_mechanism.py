"""E6 follow-up: do explanations point at the shifted features, and do the two
ECN flag features (CWR, ECE) drive the 2017 -> 2018 collapse?
A. For central and FedAvg alpha=100 models (5 seeds), per family: attribution share
   of CWR+ECE on 2017 vs 2018, best ECN rank on 2018, top-5 features, and the
   Spearman correlation between each feature's |median shift| and its change in
   attribution (positive = explanations move toward the shifted features).
B. Ablation: centralized MLP trained on 2017 WITHOUT CWR/ECE (5 seeds, 50 epochs,
   resumable) -> 2017 Test-B and 2018 per-family F1, vs the full-feature models.
Usage: python scripts/e6_mechanism.py      (~45-50 min, mostly part B)
Output: results/analysis/e6_mechanism.json + e6_mechanism_summary.txt
"""
import json, os, sys, time
import numpy as np, torch, yaml
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, class_weights, CLASSES
from fedxai.train.common import load_model, load_2018
from fedxai.train.model import MLP
from fedxai.train.fl import train_centralized
from fedxai.train.evaluate import detection_metrics
from fedxai.explain.eg import expected_gradients
from fedxai.metrics.agreement import importance, t_ci

NS, PER, EPOCHS = 1000, 200, 50
ECN = ["cwrflagcount", "eceflagcount"]
C = len(CLASSES)


def main():
    cfg = yaml.safe_load(open("configs/protocol.yaml")); od = cfg["data"]["out_dir"]; seeds = cfg["seeds"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, idx, feats = load_2017(od, dev); X8, y8, f8 = load_2018(od, dev)
    assert f8 == list(feats)
    ecn = [feats.index(f) for f in ECN if f in feats]
    yR = y[idx["R"]].cpu().numpy(); Xbg = X[idx["background"]]; tb = idx["test_b"]; ytb = y[tb]
    fams = [c for c in range(1, C) if (y8 == c).sum() >= 30 and (yR == c).sum() >= 30]
    g = torch.Generator().manual_seed(0)
    pick = lambda pool: pool[torch.randperm(len(pool), generator=g)[:PER].to(dev)]
    s17 = {c: pick(tb[ytb == c]) for c in fams}; s18 = {c: pick(torch.nonzero(y8 == c).squeeze(1)) for c in fams}
    Xtr, ytr = X[idx["train"]].cpu().numpy(), y[idx["train"]].cpu().numpy()
    X8n, y8n = X8.cpu().numpy(), y8.cpu().numpy()
    shift = {c: np.abs(np.median(X8n[y8n == c], 0) - np.median(Xtr[ytr == c], 0)) for c in fams}
    res, L, t0 = {"families": [CLASSES[c] for c in fams], "ecn_features": [feats[j] for j in ecn]}, [], time.time()

    # ---- A ----
    A = {}
    for mt, pat in {"central": "central_s{s}", "fedavg_a100": "fedavg_dir_a100_s{s}"}.items():
        A[mt] = {}
        for c in fams:
            sh17, sh18, best, rho, top17, top18 = [], [], [], [], {}, {}
            for s in seeds:
                m = load_model(os.path.join("results", "runs", pat.format(s=s), "final.pt"), X.shape[1], dev)
                i17 = importance(expected_gradients(m, X[s17[c]], Xbg, y[s17[c]], nsamples=NS, seed=1))
                i18 = importance(expected_gradients(m, X8[s18[c]], Xbg, y8[s18[c]], nsamples=NS, seed=1))
                sh17.append(float(i17[ecn].sum())); sh18.append(float(i18[ecn].sum()))
                rank18 = np.argsort(np.argsort(-i18))
                best.append(int(rank18[ecn].min()) + 1)
                rho.append(float(stats.spearmanr(shift[c], i18 - i17).statistic))
                for j in np.argsort(-i17)[:5]: top17[feats[j]] = top17.get(feats[j], 0) + 1
                for j in np.argsort(-i18)[:5]: top18[feats[j]] = top18.get(feats[j], 0) + 1
            A[mt][CLASSES[c]] = {"ecn_share_2017": t_ci(sh17), "ecn_share_2018": t_ci(sh18),
                                 "best_ecn_rank_2018_median": float(np.median(best)),
                                 "shift_vs_attribution_change_spearman": t_ci(rho),
                                 "top5_2017_counts": dict(sorted(top17.items(), key=lambda kv: -kv[1])[:6]),
                                 "top5_2018_counts": dict(sorted(top18.items(), key=lambda kv: -kv[1])[:6])}
    res["A_attribution"] = A
    print(f"[{(time.time() - t0) / 60:.1f} min] A done", flush=True)

    # ---- B ----
    keep = [j for j in range(X.shape[1]) if j not in ecn]
    Xk, X8k = X[:, keep], X8[:, keep]; w = class_weights(y[idx["train"]])
    B = {"full": [], "no_ecn": []}
    for s in seeds:
        rd = os.path.join("results", "runs", f"central_noecn_s{s}"); os.makedirs(rd, exist_ok=True)
        mp = os.path.join(rd, "metrics.json")
        if not os.path.exists(mp):
            torch.manual_seed(s); m = MLP(len(keep), C).to(dev)
            train_centralized(m, Xk, y, idx["train"], EPOCHS, w, seed=s,
                              log=lambda msg: print(msg, flush=True) if any(f"epoch {e}/50" in msg for e in (10, 20, 30, 40, 50)) else None)
            torch.save(m.state_dict(), os.path.join(rd, "final.pt"))
            json.dump({"testb": detection_metrics(m, Xk, y, tb, C),
                       "y2018": detection_metrics(m, X8k, y8, torch.arange(len(y8), device=dev), C)}, open(mp, "w"))
            print(f"[{(time.time() - t0) / 60:.1f} min] no-ECN seed {s} trained", flush=True)
        B["no_ecn"].append(json.load(open(mp)))
        mf = load_model(os.path.join("results", "runs", f"central_s{s}", "final.pt"), X.shape[1], dev)
        B["full"].append({"testb": detection_metrics(mf, X, y, tb, C),
                          "y2018": detection_metrics(mf, X8, y8, torch.arange(len(y8), device=dev), C)})
    Bs = {}
    for k, runs in B.items():
        Bs[k] = {"testb_macro_f1": t_ci([r["testb"]["macro_f1"] for r in runs]),
                 "f1_2018": {CLASSES[c]: t_ci([r["y2018"]["per_class_f1"][c] for r in runs]) for c in [0] + fams}}
    Bs["paired_diff_2018_no_ecn_minus_full"] = {
        CLASSES[c]: t_ci([B["no_ecn"][i]["y2018"]["per_class_f1"][c] - B["full"][i]["y2018"]["per_class_f1"][c]
                          for i in range(len(seeds))]) for c in [0] + fams}
    res["B_ablation"] = Bs

    json.dump(res, open("results/analysis/e6_mechanism.json", "w"), indent=1, default=float)
    L.append("=== A. Attribution share of CWR+ECE (2017 -> 2018), best ECN rank in 2018, shift-attribution rho ===")
    for mt, d in A.items():
        L.append(f"[{mt}]")
        for fn, v in d.items():
            L.append(f"  {fn:11s} share {v['ecn_share_2017'][0]:.3f} -> {v['ecn_share_2018'][0]:.3f} | "
                     f"best rank {v['best_ecn_rank_2018_median']:.0f} | rho {v['shift_vs_attribution_change_spearman'][0]:+.3f}"
                     f"+-{v['shift_vs_attribution_change_spearman'][1]:.3f} | top5 2018: {list(v['top5_2018_counts'])[:3]}")
    L.append("\n=== B. 2018 per-family F1: full features | without CWR/ECE | paired diff ===")
    L.append(f"2017 Test-B macro-F1: full {Bs['full']['testb_macro_f1'][0]:.4f} | no-ECN {Bs['no_ecn']['testb_macro_f1'][0]:.4f}")
    for cn in Bs["full"]["f1_2018"]:
        d = Bs["paired_diff_2018_no_ecn_minus_full"][cn]
        L.append(f"  {cn:11s} {Bs['full']['f1_2018'][cn][0]:.3f} | {Bs['no_ecn']['f1_2018'][cn][0]:.3f} | {d[0]:+.3f}+-{d[1]:.3f}")
    open("results/analysis/e6_mechanism_summary.txt", "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
