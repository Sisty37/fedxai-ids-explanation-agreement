"""Sanity checks before interpreting E6 (is the 2017 -> 2018 collapse real or a bug?).
1. In-domain 2018: train the same MLP on the 2018 sample (70/30 split, 10 epochs).
   High F1 => 2018 processing and labels are fine, so the collapse is a real shift.
2. Feature shift: per feature, median of 2018 vs 2017 (Benign and each family),
   in units of 2017 std after the 2017 normalisation. Many features shifted by
   orders of magnitude would point to a unit / extraction mismatch.
Usage: python scripts/sanity_e6.py      (~2 min)  Output: results/analysis/sanity_e6.txt
"""
import json, os, sys
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, class_weights, CLASSES
from fedxai.data.fed_norm import NormStats
from fedxai.train.model import MLP
from fedxai.train.fl import train_centralized
from fedxai.train.evaluate import detection_metrics


def main():
    cfg = yaml.safe_load(open("configs/protocol.yaml")); od = cfg["data"]["out_dir"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"; C = len(CLASSES)
    X, y, idx, feats = load_2017(od, dev)
    z = np.load(os.path.join(od, "cicids2018.npz"), allow_pickle=False)
    ns = json.load(open(os.path.join(od, "norm_stats_full.json")))
    st = NormStats(np.array(ns["log1p_mask"]), np.array(ns["mean"]), np.array(ns["std"]))
    X8 = torch.from_numpy(st.transform(z["X"])).to(dev)
    ci = {c: i for i, c in enumerate(CLASSES)}
    y8 = torch.from_numpy(np.array([ci[v] for v in z["y"]], dtype=np.int64)).to(dev)
    L = []

    # 1. in-domain 2018
    g = torch.Generator().manual_seed(0); perm = torch.randperm(len(y8), generator=g).to(dev)
    ntr = int(0.7 * len(perm)); tr, te = perm[:ntr], perm[ntr:]
    torch.manual_seed(0); m = MLP(X8.shape[1], C).to(dev)
    train_centralized(m, X8, y8, tr, 10, class_weights(y8[tr]), seed=0, log=lambda *_: None)
    d = detection_metrics(m, X8, y8, te, C)
    L.append("=== 1. In-domain 2018 (train and test on 2018 sample) ===")
    L += [f"{CLASSES[c]:11s} F1 {d['per_class_f1'][c]:.3f}  (support {d['support'][c]})" for c in range(C) if d["support"][c]]

    # 2. feature shift
    L.append("\n=== 2. Median shift 2018 vs 2017, in 2017-std units (after 2017 normalisation) ===")
    Xc, yc = X[idx["train"]].cpu().numpy(), y[idx["train"]].cpu().numpy()
    X8n, y8n = X8.cpu().numpy(), y8.cpu().numpy()
    for c in range(C):
        if (y8n == c).sum() < 30 or (yc == c).sum() < 30: continue
        sh = np.median(X8n[y8n == c], 0) - np.median(Xc[yc == c], 0)
        top = np.argsort(-np.abs(sh))[:5]
        L.append(f"{CLASSES[c]:11s} |shift|>1: {int((np.abs(sh) > 1).sum()):2d}/{len(sh)} features, "
                 f">5: {int((np.abs(sh) > 5).sum()):2d} | top: " + ", ".join(f"{feats[j]}({sh[j]:+.1f})" for j in top))
    raw17 = np.load(os.path.join(od, "cicids2017.npz"), allow_pickle=False)
    j = feats.index("flowduration") if "flowduration" in feats else 0
    b17 = raw17["X"][raw17["y"] == "Benign"][:, j]; b18 = z["X"][z["y"] == "Benign"][:, j]
    L.append(f"\nRaw Benign {feats[j]} median: 2017 {np.median(b17):.1f} | 2018 {np.median(b18):.1f} "
             f"(a ~1000x gap would suggest a time-unit mismatch)")
    os.makedirs("results/analysis", exist_ok=True)
    open("results/analysis/sanity_e6.txt", "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
