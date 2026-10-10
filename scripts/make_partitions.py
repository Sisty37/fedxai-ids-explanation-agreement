"""Week 1: client partitions for every (partition, seed). Test-B is split with the
same per-class client proportions (client-local held-out explanation data).

Usage: python scripts/make_partitions.py --config configs/protocol.yaml
Output: data/processed/partitions/{dir_a<alpha>|attack_skew}_s<seed>.npz
"""
import argparse, json, os, sys
import numpy as np, yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.data.partition import dirichlet_partition, attack_skew_partition, local_test_split


def save(path, train_parts, test_parts, P, classes, train_idx, test_idx):
    np.savez_compressed(path, P=P, classes=np.array(classes),
                        **{f"train_{k}": train_idx[p] for k, p in enumerate(train_parts)},
                        **{f"testb_{k}": test_idx[p] for k, p in enumerate(test_parts)})
    return {"train_sizes": [len(p) for p in train_parts], "testb_sizes": [len(p) for p in test_parts]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/protocol.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    out = cfg["data"]["out_dir"]; pdir = os.path.join(out, "partitions"); os.makedirs(pdir, exist_ok=True)
    z = np.load(os.path.join(out, "cicids2017.npz"), allow_pickle=False)
    y, tr, tb = z["y"], z["train"], z["test_b"]
    ytr, ytb = y[tr], y[tb]
    classes = list(np.unique(ytr))
    pc = cfg["partition"]; K = pc["n_clients"]; summary = {}
    for seed in cfg["seeds"]:
        for a in pc["dirichlet_alphas"]:
            parts, P = dirichlet_partition(ytr, K, a, seed, pc["min_client_size"])
            tparts = local_test_split(ytb, P, classes, seed)
            name = f"dir_a{a}_s{seed}"
            summary[name] = save(os.path.join(pdir, name + ".npz"), parts, tparts, P, classes, tr, tb)
        parts, P = attack_skew_partition(ytr, K, "Benign", seed, pc["attack_skew_families_per_client"])
        tparts = local_test_split(ytb, P, classes, seed)
        name = f"attack_skew_s{seed}"
        summary[name] = save(os.path.join(pdir, name + ".npz"), parts, tparts, P, classes, tr, tb)
    json.dump(summary, open(os.path.join(pdir, "summary.json"), "w"), indent=1)
    print(f"wrote {len(summary)} partitions to {pdir}")


if __name__ == "__main__":
    main()
