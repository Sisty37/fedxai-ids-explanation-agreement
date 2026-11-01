"""Camera-ready review item (R4 #3, R6): quantify what a client's shared explanation
summary reveals to an honest-but-curious server.

Threat model: the server holds the global model and the public reference set R (with
labels), so it can compute the model's class-wise mean |SHAP| vectors m_c on R. A client
shares a normalized importance vector s computed on its own flows.

A. Overall vector s (as used for client-level comparisons): s is close to the class-mix
   prediction sum_c pi_c m_c / Z (Sect. 5.1), so the server can invert it with non-negative
   least squares and estimate the client's class shares pi. We report the total-variation
   (TV) error of this estimate, the TV error of a server that only knows the average class
   shares over clients (prior), and the AUC for detecting which classes a client holds.
B. Class-conditional vectors (the recommended protocol): a vector is shared only for classes
   with >= 30 flows, so class presence is revealed by design, but class shares are not.
C. Gaussian noise on the shared vector (sigma in multiples of 1/F, F = number of features):
   leakage (TV error of the class-share estimate from the noisy overall vector) against
   utility loss (RBO divergence between noisy and clean class-conditional vectors).

Uses the saved FedAvg checkpoints; no training. Datasets: CICIDS2017 (main) and UNSW-NB15.
Usage:  python scripts/privacy_leakage.py            (~5-10 min on an RTX 4050)
Output: results/analysis/privacy_leakage.json + privacy_leakage_summary.txt
"""
import argparse, json, os, sys, time
import numpy as np, torch, yaml
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from fedxai.explain.eg import expected_gradients
from fedxai.metrics.agreement import importance, d_rbo, t_ci
from fedxai.train.model import MLP

NS = int(os.environ.get("FEDXAI_NS", 1000))
SIGMAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]    # noise SD in units of 1/F
ALPHAS = ["100", "1", "0.1"]


def tv(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def noisy(s, sigma, rng):
    if sigma == 0: return s
    z = np.clip(s + rng.normal(0, sigma, size=s.shape), 0, None)
    return z / z.sum() if z.sum() > 0 else np.full_like(s, 1 / len(s))


def run_dataset(tag, X, y, R, Xbg, C, n_feat, model_path, part_loader, seeds, minc, max_expl, rng):
    yR = y[R].cpu().numpy()
    present = [c for c in range(C) if (yR == c).any()]
    out = {}
    for a in ALPHAS:
        rows = []  # per client: tv_est per sigma, tv_prior, true pi, est pi, utility per sigma
        for s in seeds:
            m = MLP(X.shape[1], C).to(X.device)
            m.load_state_dict(torch.load(model_path(a, s), map_location=X.device, weights_only=True)); m.eval()
            phR = expected_gradients(m, X[R], Xbg, y[R], nsamples=NS, seed=0)
            M = np.stack([np.abs(phR[yR == c]).mean(0) for c in present], 1)        # F x C'
            tb = part_loader(a, s)
            for ci, ib in enumerate(tb):
                g = torch.Generator().manual_seed(1000 * s + ci)
                sel = ib[torch.randperm(len(ib), generator=g)[:max_expl].to(X.device)]
                phi = expected_gradients(m, X[sel], Xbg, y[sel], nsamples=NS, seed=ci)
                yk = y[sel].cpu().numpy()
                pi = np.bincount(yk, minlength=C)[present] / len(yk)
                sk = importance(phi)
                row = {"pi": pi, "est": {}, "util": {}}
                for sg in SIGMAS:
                    w, _ = nnls(M, noisy(sk, sg / n_feat, rng))
                    row["est"][sg] = w / w.sum() if w.sum() > 0 else np.full(len(present), 1 / len(present))
                    # utility: divergence between clean and noisy class-conditional vectors
                    us = [d_rbo(importance(phi[yk == c]), noisy(importance(phi[yk == c]), sg / n_feat, rng), 0.9)
                          for c in present if (yk == c).sum() >= minc]
                    row["util"][sg] = float(np.mean(us)) if us else np.nan
                rows.append(row)
        prior = np.mean([r["pi"] for r in rows], 0)
        res = {"n_clients": len(rows), "tv_prior": t_ci([tv(r["pi"], prior) for r in rows])}
        for sg in SIGMAS:
            est = np.array([r["est"][sg] for r in rows]); pis = np.array([r["pi"] for r in rows])
            lab = (pis >= 0.01).ravel().astype(int); sc = est.ravel()
            res[f"sigma_{sg}"] = {"tv_est": t_ci([tv(r["est"][sg], r["pi"]) for r in rows]),
                                  "presence_auc": float(roc_auc_score(lab, sc)) if 0 < lab.sum() < len(lab) else None,
                                  "utility_D_noisy_vs_clean": float(np.nanmean([r["util"][sg] for r in rows]))}
        # class-conditional protocol: what fraction of (client, class) cells reveal presence
        res["classcond_revealed_presence_rate"] = float(np.mean([(r["pi"] * max_expl >= minc).mean() for r in rows]))
        out[a] = res
        print(f"[{tag}] alpha={a}: TV est {res['sigma_0.0']['tv_est'][0]:.3f} vs prior {res['tv_prior'][0]:.3f}, "
              f"AUC {res['sigma_0.0']['presence_auc']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="all", choices=["all", "cicids", "unsw"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0); t0 = time.time(); res = {"sigmas_in_units_of_1_over_F": SIGMAS}
    cfg = yaml.safe_load(open("configs/protocol.yaml")); ex = cfg["explanation"]
    if a.dataset in ("all", "cicids"):
        from fedxai.train.data import load_2017, load_partition, CLASSES
        X, y, idx, feats = load_2017(cfg["data"]["out_dir"], dev)
        res["cicids2017"] = run_dataset(
            "CICIDS2017", X, y, idx["R"], X[idx["background"]], len(CLASSES), X.shape[1],
            lambda al, s: os.path.join("results", "runs", f"fedavg_dir_a{al}_s{s}", "final.pt"),
            lambda al, s: load_partition(cfg["data"]["out_dir"], f"dir_a{al}_s{s}", dev, cfg["partition"]["n_clients"])[1],
            cfg["seeds"], ex["min_class_samples"], ex["max_explained_per_client"], rng)
        del X, y; torch.cuda.empty_cache()
    if a.dataset in ("all", "unsw") and os.path.exists("data/processed/unsw/unsw.npz"):
        import unsw
        X, y, idx, feats, classes = unsw.load(dev)
        seeds = sorted({int(n.split("_s")[-1]) for n in os.listdir(os.path.join(unsw.RES, "runs")) if n.startswith("central_s")})
        res["unsw_nb15"] = run_dataset(
            "UNSW-NB15", X, y, idx["R"], X[idx["background"]], len(classes), X.shape[1],
            lambda al, s: os.path.join(unsw.RES, "runs", f"fedavg_dir_a{al}_s{s}", "final.pt"),
            lambda al, s: unsw.load_part(f"dir_a{al}_s{s}", dev)[1],
            seeds, unsw.MINC, unsw.MAX_EXPL, rng)
    os.makedirs("results/analysis", exist_ok=True)
    json.dump(res, open("results/analysis/privacy_leakage.json", "w"), indent=1, default=float)
    L = ["Class-share inference from a shared overall importance vector (TV error; lower = more leakage)"]
    for ds in ["cicids2017", "unsw_nb15"]:
        if ds not in res: continue
        L.append(f"\n=== {ds} ===")
        for al, d in res[ds].items():
            L.append(f"alpha={al:4s} prior-only TV {d['tv_prior'][0]:.3f} | class-cond. presence revealed in "
                     f"{100 * d['classcond_revealed_presence_rate']:.0f}% of cells")
            for sg in SIGMAS:
                e = d[f"sigma_{sg}"]
                L.append(f"   noise {sg:4.2f}/F: TV est {e['tv_est'][0]:.3f}+-{e['tv_est'][1]:.3f} | presence AUC "
                         f"{e['presence_auc'] if e['presence_auc'] is None else round(e['presence_auc'], 3)} | utility loss D {e['utility_D_noisy_vs_clean']:.3f}")
    open("results/analysis/privacy_leakage_summary.txt", "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L))
    print(f"[{(time.time() - t0) / 60:.1f} min] done")


if __name__ == "__main__":
    main()
