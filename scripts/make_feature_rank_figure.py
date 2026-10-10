"""Qualitative figure: top features per class across models (expected-gradients SHAP).
For each class (default DoS and Bot): the centralized model's top-10 features (rows) and,
for each model (columns), that feature's rank (number) and normalized importance (colour).
Importances are averaged over the 5 seeds before ranking, on the shared reference set R.

Usage: python scripts/make_feature_rank_figure.py [--classes DoS Bot] [--out figures]   (~2 min)
Output: figures/fig5_feature_ranks.pdf (+ feature_ranks.json with the numbers)
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5})
MODELS = [("Centralized", "central_s{s}"), ("FedAvg\nα=100", "fedavg_dir_a100_s{s}"),
          ("FedAvg\nα=0.1", "fedavg_dir_a0.1_s{s}"), ("FedProx\nα=0.1", "fedprox_mu0.01_dir_a0.1_s{s}")]
TOP = 10


def compute(classes, seeds):
    import torch, yaml
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from fedxai.train.data import load_2017, CLASSES
    from fedxai.train.common import load_model
    from fedxai.explain.eg import expected_gradients
    from fedxai.metrics.agreement import importance
    cfg = yaml.safe_load(open("configs/protocol.yaml")); dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, idx, feats = load_2017(cfg["data"]["out_dir"], dev)
    R, bg = idx["R"], idx["background"]; yR = y[R].cpu().numpy()
    out = {"features": list(map(str, feats)), "imp": {}}
    for lab, pat in MODELS:
        phis = [expected_gradients(load_model(os.path.join("results", "runs", pat.format(s=s), "final.pt"), X.shape[1], dev),
                                   X[R], X[bg], y[R], nsamples=1000, seed=0) for s in seeds]
        out["imp"][lab] = {c: np.mean([importance(p[yR == CLASSES.index(c)]) for p in phis], 0).tolist() for c in classes}
    return out


def plot(data, classes, path):
    feats = data["features"]; labs = [m for m, _ in MODELS]
    fig, axes = plt.subplots(1, len(classes), figsize=(4.8, 2.35))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, classes):
        imps = {m: np.array(data["imp"][m][c]) for m in labs}
        ranks = {m: np.argsort(np.argsort(-imps[m])) + 1 for m in labs}
        top = np.argsort(-imps[labs[0]])[:TOP]
        M = np.array([[imps[m][j] for m in labs] for j in top])
        im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=M.max())
        for i, j in enumerate(top):
            for k, m in enumerate(labs):
                r = ranks[m][j]
                ax.text(k, i, f"{r}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, k] > 0.55 * M.max() else ("#D55E00" if r > TOP else "black"),
                        weight="bold" if r > TOP else "normal")
        ax.set_xticks(range(len(labs)), labs, fontsize=5.8); ax.xaxis.tick_top()
        ax.set_yticks(range(TOP), [feats[j] for j in top], fontsize=5.8)
        ax.set_title(c, fontsize=7.5, weight="bold", pad=18)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(length=0)
    fig.text(0.5, 0.0, "Number = rank of the feature in that model (orange bold = outside its top 10); "
             "shade = normalized importance. Mean over 5 seeds, reference set R.", ha="center", fontsize=5.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=1.5)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02); print("wrote", path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--classes", nargs="+", default=["DoS", "Bot"])
    ap.add_argument("--out", default="figures"); ap.add_argument("--from-json", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.from_json:
        data = json.load(open(a.from_json))
    else:
        data = compute(a.classes, [0, 1, 2, 3, 4])
        json.dump(data, open(os.path.join(a.out, "feature_ranks.json"), "w"))
    plot(data, a.classes, os.path.join(a.out, "fig5_feature_ranks.pdf"))


if __name__ == "__main__":
    main()
