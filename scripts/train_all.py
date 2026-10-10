"""Train every model needed for E1, E2 and E3 and save checkpoints + detection
metrics. Explanations are computed later from the checkpoints (separate script),
so training never has to be repeated. Safe to stop and restart: finished runs
are skipped.

Usage (from the fedxai-ids folder):
  python scripts/train_all.py --dry-run     # list runs and estimated time
  python scripts/train_all.py               # run everything (~6 h on RTX 4050)
  python scripts/train_all.py --fedprox     # E5: 40 FedProx runs (~7 h)
Output: results/runs/<run>/{final.pt, round_<r>.pt, metrics.json}
"""
import argparse, json, os, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.train.data import load_2017, load_partition, class_weights, CLASSES
from fedxai.train.model import MLP
from fedxai.train.fl import train_centralized, train_federated
from fedxai.train.evaluate import detection_metrics

ROUNDS = 50
TRAJ_ALPHAS = {"100", "1", "0.1"}
TRAJ_ROUNDS = (1, 5, 10, 20, 30, 40, 50)
SEC_PER_PASS = 11.0  # measured in the pilot (one pass over the training data)


MU_MAIN, MU_EXTRA = 0.01, (0.001, 0.1)


def plan_runs(cfg, fedprox=False):
    runs = []
    if fedprox:  # E5: mu=0.01 at every alpha; mu in {0.001, 0.1} at alpha=0.1
        for s in cfg["seeds"]:
            for a in cfg["partition"]["dirichlet_alphas"]:
                runs.append(("fedprox", f"dir_a{a}", s, MU_MAIN))
            for mu in MU_EXTRA:
                runs.append(("fedprox", "dir_a0.1", s, mu))
        return runs
    for s in cfg["seeds"]:
        runs.append(("central", None, s, 0.0))
        for a in cfg["partition"]["dirichlet_alphas"]:
            runs.append(("fedavg", f"dir_a{a}", s, 0.0))
        runs.append(("fedavg", "attack_skew", s, 0.0))
    return runs


def run_name(kind, part, s, mu=0.0):
    if kind == "central":
        return f"central_s{s}"
    if kind == "fedprox":
        return f"fedprox_mu{mu}_{part}_s{s}"
    return f"fedavg_{part}_s{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fedprox", action="store_true", help="E5 FedProx runs instead of E1-E3")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    runs = plan_runs(cfg, a.fedprox)
    todo = [r for r in runs if not os.path.exists(os.path.join("results", "runs", run_name(*r), "metrics.json"))]
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} already done, {len(todo)} to do")
    print(f"estimated time: {len(todo) * ROUNDS * SEC_PER_PASS / 3600:.1f} h")
    if a.dry_run:
        for r in todo:
            print("  ", run_name(*r))
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["data"]["out_dir"]
    X, y, idx, _ = load_2017(out_dir, dev)
    w = class_weights(y[idx["train"]])
    C, n_clients = len(CLASSES), cfg["partition"]["n_clients"]
    t_all = time.time()
    for i, (kind, part, s, mu) in enumerate(todo, 1):
        name = run_name(kind, part, s, mu)
        rd = os.path.join("results", "runs", name); os.makedirs(rd, exist_ok=True)
        print(f"\n=== [{i}/{len(todo)}] {name} ===")
        torch.manual_seed(s); np.random.seed(s)
        model = MLP(X.shape[1], C).to(dev)
        t = time.time()
        meta = {"run": name, "seed": s, "mu": mu, "rounds_or_epochs": ROUNDS, "torch": torch.__version__}
        if kind == "central":
            # same number of passes over the data as 50 FL rounds x 1 local epoch
            train_centralized(model, X, y, idx["train"], ROUNDS, w, seed=s)
        else:
            tr, tb = load_partition(out_dir, f"{part}_s{s}", dev, n_clients)
            alpha = part.replace("dir_a", "") if part.startswith("dir_a") else None
            ck = TRAJ_ROUNDS if (alpha in TRAJ_ALPHAS and kind == "fedavg") else ()
            save = lambda r, m: torch.save(m.state_dict(), os.path.join(rd, f"round_{r}.pt"))
            train_federated(model, X, y, tr, ROUNDS, w, mu=mu, seed=s, checkpoints=ck, on_checkpoint=save)
            meta["per_client_testb"] = [detection_metrics(model, X, y, ib, C) for ib in tb]
            meta["partition"] = part
        meta["train_time_s"] = round(time.time() - t, 1)
        meta["testb"] = detection_metrics(model, X, y, idx["test_b"], C)
        torch.save(model.state_dict(), os.path.join(rd, "final.pt"))
        json.dump(meta, open(os.path.join(rd, "metrics.json"), "w"), indent=1)  # written last = done
        el = time.time() - t_all
        print(f"macro-F1 {meta['testb']['macro_f1']:.4f} | elapsed {el / 3600:.2f} h | "
              f"ETA {(len(todo) - i) * el / i / 3600:.2f} h")


if __name__ == "__main__":
    main()
