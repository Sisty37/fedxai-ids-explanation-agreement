"""Week 1: clean, dedupe, split, reference set, normalisation stats, lock record.

Usage: python scripts/prepare_data.py --config configs/protocol.yaml [--skip-2018]
Outputs (data/processed/):
  cicids2017.npz      X (float64 raw, cleaned), y (str), split idx, R / background idx
  cicids2018.npz      X, y restricted to common features (transfer test)
  features.json       final 2017 features, common 2017/2018 features
  norm_stats_*.json   federated-equivalent normalisation stats (full / common)
  lock_record.json    every removed count + final class counts -> commit this
"""
import argparse, json, os, sys, platform
import numpy as np, yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.data.prepare import load_raw, clean, stratified_split, constant_columns
from fedxai.data.fed_norm import federated_norm_stats, build_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--skip-2018", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    d, seed = cfg["data"], cfg["seed"]
    out = d["out_dir"]; os.makedirs(out, exist_ok=True)
    rec = {"config": cfg, "python": platform.python_version()}

    # ---- CICIDS2017 (training dataset) ----
    X, y, r17 = clean(load_raw(d["cicids2017_dir"], "2017"), d["include_dst_port"])
    sp = cfg["split"]
    tr, ta, tb = stratified_split(y, (sp["train"], sp["test_a"], sp["test_b"]), seed)
    const = constant_columns(X.iloc[tr])
    X = X.drop(columns=const)
    r17["columns_removed_constant_train"] = const
    feats = list(X.columns)
    yv = y.to_numpy()
    R_in_a, bg_in_a = build_reference(yv[ta], cfg["reference"]["per_class"],
                                      cfg["reference"]["background_size"], seed)
    R, bg = ta[R_in_a], ta[bg_in_a]
    assert not (set(R) & set(tr)) and not (set(R) & set(tb)), "R leakage"
    r17.update({"n_features_final": len(feats),
                "split_sizes": {"train": len(tr), "test_a": len(ta), "test_b": len(tb)},
                "class_counts_final": y.value_counts().to_dict(),
                "class_counts_train": y.iloc[tr].value_counts().to_dict(),
                "R_size": len(R), "R_class_counts": y.iloc[R].value_counts().to_dict(),
                "background_size": len(bg)})
    rec["cicids2017"] = r17
    np.savez_compressed(os.path.join(out, "cicids2017.npz"), X=X.to_numpy(), y=yv.astype(str),
                        train=tr, test_a=ta, test_b=tb, R=R, background=bg,
                        features=np.array(feats))
    # pooled stats == federated stats for ANY partition (sums are additive)
    ns = federated_norm_stats([X.iloc[tr].to_numpy()], cfg["transform"]["log1p_nonnegative"])
    json.dump({"features": feats, **ns.to_dict()}, open(os.path.join(out, "norm_stats_full.json"), "w"))

    common = feats
    if not a.skip_2018:
        bf = d.get("cicids2018_benign_fraction", 1.0)
        X8, y8, r18 = clean(load_raw(d["cicids2018_dir"], "2018", bf, seed), d["include_dst_port"])
        r18["benign_fraction_sampled"] = bf
        common = [f for f in feats if f in set(X8.columns)]
        r18["features_missing_in_2018"] = [f for f in feats if f not in set(X8.columns)]
        r18["features_only_in_2018"] = [f for f in X8.columns if f not in set(feats)]
        r18["class_counts_final"] = y8.value_counts().to_dict()
        rec["csecicids2018"] = r18
        np.savez_compressed(os.path.join(out, "cicids2018.npz"), X=X8[common].to_numpy(),
                            y=y8.to_numpy().astype(str), features=np.array(common))
        ci = [feats.index(f) for f in common]
        nsc = federated_norm_stats([X.iloc[tr].to_numpy()[:, ci]], cfg["transform"]["log1p_nonnegative"])
        json.dump({"features": common, **nsc.to_dict()}, open(os.path.join(out, "norm_stats_common.json"), "w"))
    json.dump({"full_2017": feats, "common_2017_2018": common},
              open(os.path.join(out, "features.json"), "w"), indent=1)
    json.dump(rec, open(os.path.join(out, "lock_record.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in r17.items() if k.startswith(("rows", "split", "n_feat"))}, indent=1))


if __name__ == "__main__":
    main()
