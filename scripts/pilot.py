"""Week 2 pilot: centralised MLP + one FedAvg run + expected-gradients explanations.
Checks that everything runs on the real data, measures time, and tests the
class-mixture effect on client-level explanations.

Usage (from the fedxai-ids folder):
  python scripts/pilot.py                       # alpha=1, seed 0, 10 rounds
  python scripts/pilot.py --rounds 50           # full-length run
Output: results/pilot/pilot.json (+ model checkpoints)
"""
import argparse, json, os, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, load_partition, class_weights, CLASSES
from fedxai.train.model import MLP
from fedxai.train.fl import train_centralized, train_federated
from fedxai.train.evaluate import detection_metrics
from fedxai.explain.eg import expected_gradients
from fedxai.metrics.agreement import importance, d_rbo, client_eed

C = len(CLASSES)


def explain(model, X, y, idx, bg, seed):
    return expected_gradients(model, X[idx], X[bg], y[idx], nsamples=200, seed=seed)


def class_means(phi, yy):
    """Unnormalised mean |phi| per class -> {c: (F,)}"""
    return {c: np.abs(phi[yy == c]).mean(0) for c in range(C) if (yy == c).any()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--alpha", default="1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--central-epochs", type=int, default=3)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ex = cfg["explanation"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join("results", "pilot"); os.makedirs(out, exist_ok=True)
    res = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
           "torch": torch.__version__, "args": vars(a), "timing_s": {}}
    print("device:", res["device"])

    X, y, idx, feats = load_2017(cfg["data"]["out_dir"], dev)
    w = class_weights(y[idx["train"]])
    res["class_weights"] = dict(zip(CLASSES, [round(v, 3) for v in w.tolist()]))
    R, bg = idx["R"], idx["background"]
    yR = y[R].cpu().numpy()

    # ---- E1: centralised reference ----
    torch.manual_seed(a.seed)
    cen = MLP(X.shape[1], C).to(dev)
    t = time.time()
    train_centralized(cen, X, y, idx["train"], a.central_epochs, w, seed=a.seed)
    res["timing_s"]["central_train"] = round(time.time() - t, 1)
    res["central"] = detection_metrics(cen, X, y, idx["test_b"], C)
    torch.save(cen.state_dict(), os.path.join(out, "central.pt"))

    # ---- FedAvg ----
    tr, tb = load_partition(cfg["data"]["out_dir"], f"dir_a{a.alpha}_s{a.seed}", dev)
    torch.manual_seed(a.seed)
    fed = MLP(X.shape[1], C).to(dev)
    t = time.time()
    train_federated(fed, X, y, tr, a.rounds, w, seed=a.seed)
    res["timing_s"]["fed_train"] = round(time.time() - t, 1)
    res["timing_s"]["fed_per_round"] = round(res["timing_s"]["fed_train"] / a.rounds, 1)
    res["fedavg"] = detection_metrics(fed, X, y, idx["test_b"], C)
    torch.save(fed.state_dict(), os.path.join(out, f"fedavg_a{a.alpha}_s{a.seed}_r{a.rounds}.pt"))

    # ---- explanations on R ----
    t = time.time()
    phi_c = explain(cen, X, y, R, bg, a.seed)
    res["timing_s"]["eg_R_1052"] = round(time.time() - t, 1)
    phi_f = explain(fed, X, y, R, bg, a.seed)

    # completeness sanity check: sum phi ~ f_t(x) - E f_t(bg)
    with torch.no_grad():
        fed.eval()
        fx = fed(X[R]).gather(1, y[R].unsqueeze(1)).squeeze(1)
        fb = fed(X[bg])[:, y[R]].mean(0)
        gap = (fx - fb).cpu().numpy()
    res["eg_completeness_rel_err"] = float(np.abs(phi_f.sum(1) - gap).mean() / np.abs(gap).mean())

    # per-class fidelity: federated vs centralised on R (noise-floor corrected)
    rng = np.random.default_rng(a.seed)
    fid = {}
    for c in range(C):
        m = yR == c
        if m.sum() < ex["min_class_samples"]:
            fid[CLASSES[c]] = "insufficient samples"; continue
        r = client_eed(phi_f[m], phi_c[m], p=ex["rbo_p"], B=ex["bootstrap_B"],
                       n_pairs=ex["noise_pairs"], rng=rng)
        fid[CLASSES[c]] = {k: round(r[k], 4) if isinstance(r[k], float) else r[k]
                           for k in ["D", "floor", "EED", "significant", "n_explained"]}
    res["fidelity_fed_vs_central_on_R"] = fid

    # ---- class-mixture check on clients (same global model) ----
    mR = class_means(phi_f, yR)
    sR = importance(phi_f)
    mix = []
    t = time.time()
    for k, ib in enumerate(tb):
        g = torch.Generator().manual_seed(a.seed + k)
        sel = ib[torch.randperm(len(ib), generator=g)[:ex["max_explained_per_client"]].to(ib.device)]
        phi_k = explain(fed, X, y, sel, bg, a.seed + k)
        yk = y[sel].cpu().numpy()
        pi = np.bincount(yk, minlength=C) / len(yk)
        pred = sum(pi[c] * mR[c] for c in mR)          # mixture prediction
        sk = importance(phi_k)
        mix.append({"client": k, "n": int(len(yk)),
                    "class_share": dict(zip(CLASSES, np.round(pi, 4).tolist())),
                    "D_vs_R": round(d_rbo(sk, sR, ex["rbo_p"]), 4),
                    "D_vs_mixture_prediction": round(d_rbo(sk, pred / pred.sum(), ex["rbo_p"]), 4)})
    res["timing_s"]["eg_clients"] = round(time.time() - t, 1)
    res["client_mixture_check"] = mix
    res["summary"] = {
        "central_macro_f1": round(res["central"]["macro_f1"], 4),
        "fedavg_macro_f1": round(res["fedavg"]["macro_f1"], 4),
        "mean_D_client_vs_R": round(float(np.mean([m["D_vs_R"] for m in mix])), 4),
        "mean_D_client_vs_mixture": round(float(np.mean([m["D_vs_mixture_prediction"] for m in mix])), 4)}
    json.dump(res, open(os.path.join(out, "pilot.json"), "w"), indent=1)
    print(json.dumps({"summary": res["summary"], "timing_s": res["timing_s"],
                      "eg_completeness_rel_err": round(res["eg_completeness_rel_err"], 4)}, indent=1))


if __name__ == "__main__":
    main()
