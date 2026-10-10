"""Extend the protocol from 5 to 10 seeds (review item). Backs up the config first.
Usage: python scripts/extend_seeds.py          (sets seeds: [0..9])
       python scripts/extend_seeds.py --undo   (restores the 5-seed config)
"""
import argparse, shutil, yaml

P, BAK = "configs/protocol.yaml", "configs/protocol_5seeds_backup.yaml"
ap = argparse.ArgumentParser(); ap.add_argument("--undo", action="store_true"); a = ap.parse_args()
if a.undo:
    shutil.copy(BAK, P); print("restored 5-seed config")
else:
    shutil.copy(P, BAK)
    txt = open(P).read()
    old = "seeds: [0, 1, 2, 3, 4]"
    assert old in txt, "seeds line not found (already extended?)"
    open(P, "w").write(txt.replace(old, "seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"))
    print("seeds ->", yaml.safe_load(open(P))["seeds"], "| backup:", BAK)
