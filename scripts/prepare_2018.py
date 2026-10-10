"""E6 step 1: stream CSE-CIC-IDS2018 (improved, ~36 GB) and keep a uniform random
sample of at most CAP rows per family (exact reservoir via random keys), so it fits
in 16 GB RAM. Same cleaning rules as 2017: labels mapped, Attempted/Infiltration
dropped, ID columns + Dst Port removed, +-Inf -> NaN -> row removed, exact
duplicates and label conflicts removed (within the sample). Features = the final
2017 feature list (all present in 2018).

Usage: python scripts/prepare_2018.py          (~30-60 min, mostly CSV parsing)
Output: data/processed/cicids2018.npz, data/processed/lock_record_2018.json
"""
import collections, glob, json, os, sys, time
import numpy as np, pandas as pd, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.data.schema import canonicalize_columns, map_label, DROP, LABEL

CAP, CHUNK, SEED = 100_000, 500_000, 0


def main():
    cfg = yaml.safe_load(open("configs/protocol.yaml"))
    out = cfg["data"]["out_dir"]
    feats = json.load(open(os.path.join(out, "features.json")))["full_2017"]
    files = sorted(glob.glob(os.path.join(cfg["data"]["cicids2018_dir"], "*.csv")))
    if not files: raise FileNotFoundError(cfg["data"]["cicids2018_dir"])
    rng = np.random.default_rng(SEED)
    raw_counts, rec = collections.Counter(), collections.Counter()
    kept = {}  # family -> DataFrame (features + _key)
    t0 = time.time()
    for fi, f in enumerate(files, 1):
        for df in pd.read_csv(f, chunksize=CHUNK, low_memory=False, encoding="latin-1"):
            df = canonicalize_columns(df, "2018")
            lab = df[LABEL].astype(str).str.strip()
            hdr = lab.str.lower() == "label"; rec["rows_removed_header_repeat"] += int(hdr.sum())
            df, lab = df.loc[~hdr], lab.loc[~hdr]
            raw_counts.update(lab.value_counts().to_dict())
            fam = lab.map(map_label)
            keep = fam != DROP; rec["rows_removed_dropped_classes"] += int((~keep).sum())
            missing = [c for c in feats if c not in df.columns]
            if missing: raise KeyError(f"2017 features missing in {f}: {missing}")
            X = df.loc[keep, feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            fam = fam.loc[keep]
            ok = ~X.isna().any(axis=1); rec["rows_removed_inf_nan"] += int((~ok).sum())
            X, fam = X.loc[ok].astype(np.float64), fam.loc[ok]
            rec["rows_valid"] += len(X)
            X["_key"] = rng.random(len(X))
            for g, sub in X.groupby(fam.values):
                cur = pd.concat([kept[g], sub]) if g in kept else sub
                kept[g] = cur.nsmallest(CAP, "_key") if len(cur) > CAP else cur
        print(f"[{fi}/{len(files)}] {os.path.basename(f)} | {(time.time() - t0) / 60:.1f} min | "
              f"kept {sum(len(v) for v in kept.values()):,}", flush=True)
    rec["valid_rows_per_family_before_cap"] = None  # filled below
    X = pd.concat([v.drop(columns="_key").assign(_y=g) for g, v in kept.items()], ignore_index=True)
    dup = X.duplicated(); rec["rows_removed_duplicates_in_sample"] = int(dup.sum()); X = X.loc[~dup]
    conf = X.drop(columns="_y").duplicated(keep=False)
    rec["rows_removed_label_conflicts_in_sample"] = int(conf.sum()); X = X.loc[~conf]
    y = X.pop("_y").to_numpy().astype(str)
    np.savez_compressed(os.path.join(out, "cicids2018.npz"), X=X.to_numpy(), y=y, features=np.array(feats))
    rec = dict(rec); rec.pop("valid_rows_per_family_before_cap")
    rec.update({"cap_per_family": CAP, "seed": SEED, "files": [os.path.basename(f) for f in files],
                "raw_label_counts": dict(raw_counts), "rows_final": int(len(y)),
                "class_counts_final": pd.Series(y).value_counts().to_dict(),
                "dataset_version": "distrinet_improved_cns2022"})
    json.dump(rec, open(os.path.join(out, "lock_record_2018.json"), "w"), indent=1, default=int)
    print(json.dumps({k: rec[k] for k in ["rows_final", "class_counts_final"]}, indent=1))


if __name__ == "__main__":
    main()
