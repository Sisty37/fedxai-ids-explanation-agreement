"""E4: local-only models (each client trains alone on its own data), alpha in {1, 0.1},
5 seeds x 10 clients = 100 standalone trainings, 50 epochs each (same budget as FL).
Resumable. Usage: python scripts/train_local.py [--dry-run]     (~1.5-2 h)
Output: results/local/dir_a<alpha>_s<seed>_c<k>.pt and metrics.json
"""
import argparse, json, os, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, load_partition, class_weights, CLASSES
from fedxai.train.model import MLP
from fedxai.train.fl import train_centralized
from fedxai.train.evaluate import detection_metrics

ALPHAS, EPOCHS = ["1", "0.1"], 50


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/protocol.yaml"))
    K = cfg["partition"]["n_clients"]; od = os.path.join("results", "local"); os.makedirs(od, exist_ok=True)
    mpath = os.path.join(od, "metrics.json")
    done = json.load(open(mpath)) if os.path.exists(mpath) else {}
    todo = [(al, s, k) for al in ALPHAS for s in cfg["seeds"] for k in range(K) if f"dir_a{al}_s{s}_c{k}" not in done]
    print(f"{len(todo)} local trainings to do (of {len(ALPHAS) * len(cfg['seeds']) * K}); est. {len(todo) / K * 0.17:.1f} h")
    if a.dry_run: return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, idx, _ = load_2017(cfg["data"]["out_dir"], dev)
    w = class_weights(y[idx["train"]]); C = len(CLASSES); t0 = time.time()
    cache = {}
    for i, (al, s, k) in enumerate(todo, 1):
        key = f"dir_a{al}_s{s}"
        if key not in cache: cache = {key: load_partition(cfg["data"]["out_dir"], key, dev, K)}
        tr, tb = cache[key]
        torch.manual_seed(1000 * s + k)
        m = MLP(X.shape[1], C).to(dev)
        train_centralized(m, X, y, tr[k], EPOCHS, w, seed=1000 * s + k, log=lambda *_: None)
        torch.save(m.state_dict(), os.path.join(od, f"{key}_c{k}.pt"))
        done[f"{key}_c{k}"] = {"n_train": int(len(tr[k])),
                               "train_class_counts": torch.bincount(y[tr[k]], minlength=C).tolist(),
                               "own_testb": detection_metrics(m, X, y, tb[k], C)}
        json.dump(done, open(mpath, "w"), indent=1)
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {key}_c{k} n={len(tr[k])} | {el / 60:.1f} min | ETA {(len(todo) - i) * el / i / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
