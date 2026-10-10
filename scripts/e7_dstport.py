"""E7: Destination Port sensitivity. Rebuild CICIDS2017 WITH Destination Port (same
cleaning, same split seed), train centralized MLPs (5 seeds, 50 epochs, resumable),
and compare with the main protocol: macro-F1 / per-class F1, how many rows survive
dedupe (port scans are no longer identical once the port is kept), and how much
attribution Destination Port receives.
Usage: python scripts/e7_dstport.py       (~50-60 min)
Output: results/analysis/e7_dstport.json + e7_dstport_summary.txt
"""
import json, os, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.data.prepare import load_raw, clean, stratified_split, constant_columns
from fedxai.data.fed_norm import federated_norm_stats, build_reference
from fedxai.data.schema import DST_PORT
from fedxai.train.data import class_weights, CLASSES
from fedxai.train.model import MLP
from fedxai.train.fl import train_centralized
from fedxai.train.evaluate import detection_metrics
from fedxai.explain.eg import expected_gradients
from fedxai.metrics.agreement import importance, t_ci

EPOCHS, NS = 50, 1000
C = len(CLASSES)


def main():
    cfg = yaml.safe_load(open("configs/protocol.yaml")); d = cfg["data"]; seed0 = cfg["seed"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    Xdf, ydf, rep = clean(load_raw(d["cicids2017_dir"], "2017"), include_dst_port=True)
    sp = cfg["split"]; tr, ta, tb = stratified_split(ydf, (sp["train"], sp["test_a"], sp["test_b"]), seed0)
    Xdf = Xdf.drop(columns=constant_columns(Xdf.iloc[tr])); feats = list(Xdf.columns); jp = feats.index(DST_PORT)
    ns = federated_norm_stats([Xdf.iloc[tr].to_numpy()], cfg["transform"]["log1p_nonnegative"])
    X = torch.from_numpy(ns.transform(Xdf.to_numpy())).to(dev)
    ci = {c: i for i, c in enumerate(CLASSES)}
    yv = np.array([ci[v] for v in ydf], dtype=np.int64); y = torch.from_numpy(yv).to(dev)
    Rin, bgin = build_reference(yv[ta], cfg["reference"]["per_class"], cfg["reference"]["background_size"], seed0)
    R, bg = torch.from_numpy(ta[Rin]).to(dev), torch.from_numpy(ta[bgin]).to(dev)
    trt, tbt = torch.from_numpy(tr).to(dev), torch.from_numpy(tb).to(dev)
    print(f"[{(time.time() - t0) / 60:.1f} min] data ready: {len(yv):,} rows, {len(feats)} features", flush=True)
    res = {"rows_final_with_port": int(len(yv)), "rows_removed_duplicates_with_port": rep["rows_removed_duplicates"],
           "class_counts_with_port": {CLASSES[c]: int((yv == c).sum()) for c in range(C)}, "seeds": {}}
    w = class_weights(y[trt]); yR = yv[ta][Rin]
    for s in cfg["seeds"]:
        rd = os.path.join("results", "runs", f"central_dstport_s{s}"); os.makedirs(rd, exist_ok=True)
        torch.manual_seed(s); m = MLP(X.shape[1], C).to(dev)
        if os.path.exists(os.path.join(rd, "final.pt")):
            m.load_state_dict(torch.load(os.path.join(rd, "final.pt"), map_location=dev, weights_only=True))
        else:
            train_centralized(m, X, y, trt, EPOCHS, w, seed=s, log=lambda *_: None)
            torch.save(m.state_dict(), os.path.join(rd, "final.pt"))
        m.eval()
        phi = expected_gradients(m, X[R], X[bg], y[R], nsamples=NS, seed=0)
        port = {}
        for c in range(C):
            mk = yR == c
            if mk.sum() < 30: continue
            imp = importance(phi[mk]); rank = int(np.argsort(np.argsort(-imp))[jp]) + 1
            port[CLASSES[c]] = {"share": float(imp[jp]), "rank": rank}
        res["seeds"][s] = {"testb": detection_metrics(m, X, y, tbt, C), "dst_port_attribution": port}
        print(f"[{(time.time() - t0) / 60:.1f} min] seed {s} done | macro-F1 {res['seeds'][s]['testb']['macro_f1']:.4f}", flush=True)
    main_f1 = [json.load(open(os.path.join("results", "runs", f"central_s{s}", "metrics.json")))["testb"] for s in cfg["seeds"]]
    S = res["seeds"]
    res["summary"] = {
        "macro_f1_main": t_ci([m_["macro_f1"] for m_ in main_f1]),
        "macro_f1_with_port": t_ci([S[s]["testb"]["macro_f1"] for s in S]),
        "per_class_f1_main": {CLASSES[c]: t_ci([m_["per_class_f1"][c] for m_ in main_f1]) for c in range(C)},
        "per_class_f1_with_port": {CLASSES[c]: t_ci([S[s]["testb"]["per_class_f1"][c] for s in S]) for c in range(C)},
        "dst_port_share": {cn: t_ci([S[s]["dst_port_attribution"][cn]["share"] for s in S if cn in S[s]["dst_port_attribution"]])
                           for cn in S[cfg["seeds"][0]]["dst_port_attribution"]},
        "dst_port_rank_median": {cn: float(np.median([S[s]["dst_port_attribution"][cn]["rank"] for s in S if cn in S[s]["dst_port_attribution"]]))
                                 for cn in S[cfg["seeds"][0]]["dst_port_attribution"]}}
    json.dump(res, open("results/analysis/e7_dstport.json", "w"), indent=1, default=float)
    sm = res["summary"]
    L = [f"Rows after cleaning WITH port: {res['rows_final_with_port']:,} (duplicates removed: {res['rows_removed_duplicates_with_port']:,})",
         f"PortScan rows with port: {res['class_counts_with_port']['PortScan']:,}",
         f"Macro-F1 main (no port): {sm['macro_f1_main'][0]:.4f}+-{sm['macro_f1_main'][1]:.4f} | with port: {sm['macro_f1_with_port'][0]:.4f}+-{sm['macro_f1_with_port'][1]:.4f}",
         "Per-class F1 main | with port | Dst Port attribution share (median rank):"]
    for c in range(C):
        cn = CLASSES[c]; extra = f" | {sm['dst_port_share'][cn][0]:.3f} (rank {sm['dst_port_rank_median'][cn]:.0f})" if cn in sm["dst_port_share"] else ""
        L.append(f"  {cn:11s} {sm['per_class_f1_main'][cn][0]:.3f} | {sm['per_class_f1_with_port'][cn][0]:.3f}{extra}")
    open("results/analysis/e7_dstport_summary.txt", "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
