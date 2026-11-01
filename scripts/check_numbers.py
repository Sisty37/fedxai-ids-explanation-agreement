"""Check every number quoted in the camera-ready paper against the result files.
Each line prints PASS when the value in the paper equals the value in the result file
after rounding to the number of decimals used in the paper, and FAIL otherwise.

Usage (from the repository root):  python scripts/check_numbers.py
Needs: results/analysis/*.json, results/unsw/unsw_analysis.json and, if present,
data/processed/lock_record.json and data/processed/unsw/lock_record_unsw.json.
"""
import json, os, sys
import numpy as np
from scipy import stats

R = lambda *p: json.load(open(os.path.join(*p), encoding="utf-8"))
A = lambda n: R("results", "analysis", n)
an, ex, fr, e6, em, e7, dp = (A("analysis.json"), A("extra.json"), A("fedprox_rq2.json"), A("e6.json"),
                               A("e6_mechanism.json"), A("e7_ablations.json"), A("e7_dstport.json"))
cnn, se = A("cnn_validation.json"), A("sensitivity.json")
un = R("results", "unsw", "unsw_analysis.json")
lock = R("data", "processed", "lock_record.json") if os.path.exists("data/processed/lock_record.json") else None
lku = R("data", "processed", "unsw", "lock_record_unsw.json") if os.path.exists("data/processed/unsw/lock_record_unsw.json") else None

CL = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "Bot"]
n_pass = n_fail = 0


def chk(where, what, paper, values, dec):
    """paper: list of numbers as printed in the paper; values: same-length list from the files."""
    global n_pass, n_fail
    got = [round(float(v), dec) for v in values]
    ok = all(abs(g - p) < 10 ** (-dec) / 2 + 1e-9 for g, p in zip(got, paper)) and len(got) == len(paper)
    n_pass += ok; n_fail += (not ok)
    fmt = lambda xs: ", ".join(f"{x:.{dec}f}" for x in xs)
    print(f"{'PASS' if ok else 'FAIL'}  {where:14s} {what:58s} paper [{fmt(paper)}]  file [{fmt(got)}]")


mm = lambda v: [min(v), max(v)]
tci = lambda v: (np.mean(v), stats.t.ppf(.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v)))

print("=== Setup ===")
if lock:
    c = lock["cicids2017"]
    chk("Sect. 4", "removed: attempted/infiltration/heartbleed, inf, dups, conflicts",
        [83793, 5, 316117, 28], [c["rows_removed_dropped_classes"], c["rows_removed_inf_nan"], c["rows_removed_duplicates"], c["rows_removed_label_conflicts"]], 0)
    chk("Sect. 4", "final flows, features", [1700033, 80], [c["rows_final"], c["n_features_final"]], 0)
if lku:
    chk("Sect. 4", "UNSW raw, duplicates, conflicts, final, features",
        [257673, 94928, 11506, 151239, 64], [lku["rows_raw"], lku["rows_removed_duplicates"], lku["rows_removed_label_conflicts"], lku["rows_final"], lku["n_features"]], 0)
    chk("Sect. 4", "UNSW Worms reference flows", [16], [lku["R_class_counts"]["Worms"]], 0)

print("=== Table 2 ===")
s7 = se["S7_per_class"]; C7 = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot"]
chk("Table 2", "test flows", [284763, 34327, 19029, 334, 1387, 20, 147], [s7["test_support"][c] for c in C7], 0)
chk("Table 2", "R flows", [200, 200, 200, 167, 200, 11, 74], [s7["R_counts"][c] for c in C7], 0)
for key, row in [("central", [1.000, 1.000, 1.000, 0.963, 0.996, 0.821, 0.983]), ("fedavg_a100", [1.000, 0.999, 1.000, 0.973, 0.992, 0.607, 0.959]),
                 ("fedavg_a0.1", [0.975, 0.709, 0.833, 0.454, 0.790, 0.314, 0.463])]:
    chk("Table 2", f"F1 {key}", row, [s7[key]["per_class_f1"][c][0] for c in C7], 3)

print("=== Results (CICIDS2017) ===")
D = an["D_mixture"]
chk("Abs/5.1", "attack mix explains 82--93%", [82, 93], mm([d["share_explained_by_mix"] * 100 for d in D.values()]), 0)
chk("5.1", "client vs reference D 0.33--0.54", [0.33, 0.54], mm([d["D_client_vs_R"][0] for d in D.values()]), 2)
chk("5.1", "client vs class-mix 0.04--0.06", [0.04, 0.06], mm([d["D_client_vs_mixture"][0] for d in D.values()]), 2)
bg = e7["2_background"]["1"]["local_vs_shared_same_samples"]
chk("Abs/5.1", "local background: Benign 0.07; attacks 0.30--0.64", [0.07, 0.30, 0.64], [bg["Benign"][0]] + mm([v[0] for k, v in bg.items() if k != "Benign"]), 2)
sb = an["A_seed_baseline"]
chk("5.2", "seed baseline Benign 0.02; attacks 0.13--0.28", [0.02, 0.13, 0.28], [sb["Benign"]["mean"]] + mm([v["mean"] for k, v in sb.items() if k != "Benign"]), 2)
G = ex["G_factorized"]
chk("5.2", "model term 0.45--0.75 (a=1); 0.55--0.71 (a=0.1)", [0.45, 0.75, 0.55, 0.71], mm([G["1"]["model"][c][0][0] for c in CL]) + mm([G["0.1"]["model"][c][0][0] for c in CL]), 2)
chk("5.2", "data term at most 0.04", [0.04], [max(G[a]["data"][c][0][0] for a in ["1", "0.1"] for c in CL)], 2)
chk("5.2", "interaction within +-0.04", [0.04], [max(abs(G[a]["interaction"][c][0][0]) for a in ["1", "0.1"] for c in CL)], 2)
B100 = an["B_fidelity"]["dir_a100"]
chk("5.2", "federation excess PortScan 0.18, Bot 0.33", [0.18, 0.33], [B100["PortScan"]["excess_over_seed_baseline"], B100["Bot"]["excess_over_seed_baseline"]], 2)
tr = an["E_trajectory"]["100"]
chk("5.2", "excess round 1 = 0.49, round 30 = 0.24", [0.49, 0.24], [tr["1"]["fidelity_excess_mean_over_classes"][0], tr["30"]["fidelity_excess_mean_over_classes"][0]], 2)
C = an["C_detection"]
chk("5.2", "macro-F1 FedAvg a=100 vs central", [0.933, 0.966], [C["dir_a100"]["macro_f1"], C["central"]["macro_f1"]], 3)
on = fr["I_coupling"]["onset"]
chk("5.3", "drift a=5, a=0.3, a=0.1", [0.002, 0.111, 0.276], [on[a]["explanation_drift"][0] for a in ["5", "0.3", "0.1"]], 3)
chk("5.3", "drift CI a=5, a=0.3, a=0.1", [0.012, 0.045, 0.041], [on[a]["explanation_drift"][1] for a in ["5", "0.3", "0.1"]], 3)
chk("5.3", "macro-F1 a=0.1 0.648 +- 0.095", [0.648, 0.095], [C["dir_a0.1"]["macro_f1"], C["dir_a0.1"]["ci"]], 3)
I = fr["I_coupling"]
chk("5.3", "pooled rho 0.62 [0.43, 0.75]", [0.62, 0.43, 0.75], [I["pooled_spearman"]["rho"]] + list(I["pooled_spearman"]["ci"]), 2)
chk("5.3", "within-alpha rho 0.23 [-0.06, 0.48]", [0.23, -0.06, 0.48], [I["within_alpha_spearman"]["rho"]] + list(I["within_alpha_spearman"]["ci"]), 2)
H = fr["H_fedprox"]
chk("5.4", "FedProx a=0.1: 0.648 -> 0.800 (+0.152)", [0.648, 0.800, 0.152], [H["detection"]["0.1"]["fedavg"][0], H["detection"]["0.1"]["fedprox"][0], H["detection"]["0.1"]["paired_diff"][0]], 3)
mu = H["mu_sweep_alpha0.1"]
chk("5.4", "mu sweep fidelity 0.32 / 0.45 / 0.51", [0.32, 0.45, 0.51], [np.mean([mu[m]["fidelity"][c][0] for c in CL]) for m in ["0.001", "0.01", "0.1"]], 2)
chk("5.4", "mu sweep macro-F1 0.76 / 0.80 / 0.74", [0.76, 0.80, 0.74], [mu[m]["macro_f1"][0] for m in ["0.001", "0.01", "0.1"]], 2)
cells = [c for v in e6["models"].values() for c in v["cells"]]
chk("5.5", "240 cells, all above floor; EED 0.23--0.84", [240, 240], [len(cells), sum(c["significant"] for c in cells)], 0)
chk("5.5", "EED range 0.23--0.84", [0.23, 0.84], mm([c["dE"] for c in cells]), 2)
Am = em["A_attribution"]
chk("5.5", "CWR+ECE share 88% (central), 80% (FedAvg)", [88, 80],
    [100 * np.mean([Am["central"][f]["ecn_share_2018"][0] for f in ["DDoS", "Bot"]]), 100 * np.mean([Am["fedavg_a100"][f]["ecn_share_2018"][0] for f in ["DDoS", "Bot"]])], 0)
Bm = em["B_ablation"]
chk("5.5", "no-CWR/ECE 2017 F1 0.969 vs 0.966", [0.969, 0.966], [Bm["no_ecn"]["testb_macro_f1"][0], Bm["full"]["testb_macro_f1"][0]], 3)
pdif = Bm["paired_diff_2018_no_ecn_minus_full"]
chk("5.5", "2018 F1 gain DDoS 0.22+-0.08, Bot 0.29+-0.28", [0.22, 0.08, 0.29, 0.28], [pdif["DDoS"][0], pdif["DDoS"][1], pdif["Bot"][0], pdif["Bot"][1]], 2)
k3 = e7["3_kernelshap_vs_eg"]
chk("5.6", "RBO vs J* rho 0.76, 4,680 pairs", [0.76, 4680], [e7["1_metric_check"]["rbo_vs_jstar10_spearman"], e7["1_metric_check"]["n_pairs"]], 2)
chk("5.6", "KernelSHAP overall 0.91, per class 0.36--0.78", [0.91, 0.36, 0.78], [k3["overall_spearman"]] + mm([v["spearman"] for v in k3["per_class"].values()]), 2)
k5 = se["S5_kernelshap"]["kernelshap"]
chk("5.6", "KernelSHAP fed. excess 0.06--0.29, drift 0.22--0.33", [0.06, 0.29, 0.22, 0.33], mm([k5["federation_excess"][c][0] for c in CL]) + mm([k5["drift_alpha0.1"][c][0] for c in CL]), 2)
chk("5.6", "dst port macro-F1 0.960 -> 0.966", [0.960, 0.966], [dp["summary"]["macro_f1_main"][0], dp["summary"]["macro_f1_with_port"][0]], 3)

print("=== Sensitivity ===")
S1 = se["S1_rbo_p"]
chk("5.6", "all 6 classes significant (excess, drift) for every p", [6, 6], [min(S1[p]["n_significant_fed_excess"] for p in S1), min(S1[p]["n_significant_drift"] for p in S1)], 0)
chk("5.6", "class-avg excess 0.12--0.26 across p", [0.12, 0.26], mm([S1[p]["federation_excess_classavg"][0] for p in ["0.8", "0.9", "0.95", "0.98"]]), 2)
S2 = se["S2_background"]; sizes = ["m25", "m50", "main", "m200"]
chk("5.6", "class-avg excess 0.24--0.36 across sizes", [0.24, 0.36], mm([S2[m]["federation_excess_classavg"][0] for m in sizes]), 2)
chk("5.6", "sizes: all classes significant", [1], [all(v["holm_p"] < .05 for m in sizes for k in ["federation_excess", "drift_alpha0.1"] for v in S2[m][k].values())], 0)
chk("5.6", "attack mix 83--92% across p", [83, 92], mm([S1[p]["class_mix"][a]["share_explained_by_mix"] * 100 for p in S1 for a in ["1", "0.1"]]), 0)
S4 = se["S4_bootstrap"]
chk("5.6", "floor MC SD at most 0.002 (B=1000, 5k pairs)", [0.002], [max(S4[c]["B1000_pairs5000"]["sd"] for c in CL)], 3)
chk("5.6", "floor moves < 0.002 between B=200 and B=2000", [0.001], [max(abs(S4[c]["B200_pairs5000"]["mean"] - S4[c]["B2000_pairs5000"]["mean"]) for c in CL)], 3)
S3 = se["S3_n_explained"]["by_n"]
chk("5.6", "floor with 10--20 flows 0.10--0.21", [0.10, 0.21], mm([S3[c][n]["floor_mean"] for c in CL for n in ["10", "20"]]), 2)
chk("5.6", "seed-pair EED SD up to 0.11 (10--20 flows)", [0.11], [max(S3[c][n]["eed_sd"] for c in CL for n in ["10", "20"])], 2)
chk("5.6", "two shared backgrounds: EED up to 0.26 (DDoS)", [0.26], [S2["same_model_two_shared_backgrounds_EED"]["m100_drawB"]["DDoS"][0]], 2)

print("=== Replication (Table 3, Sect. 5.7) ===")
chk("Table 3", "CNN macro-F1 central / a=0.1 (10 seeds)", [0.963, 0.498], [cnn["detection"]["central"][0], cnn["detection"]["fedavg_a0.1"][0]], 3)
chk("Table 3", "UNSW macro-F1 central / a=0.1", [0.617, 0.328], [un["C_detection"]["central"]["macro_f1"][0], un["C_detection"]["fedavg_a0.1"]["macro_f1"][0]], 3)
chk("Table 3", "CNN attack-mix share 78--81%", [78, 81], mm([d["share_explained_by_mix"] * 100 for d in cnn["class_mix"].values()]), 0)
chk("Table 3", "UNSW attack-mix share 69--83%", [69, 83], mm([d["share_explained_by_mix"] * 100 for d in un["D_mixture"].values()]), 0)
chk("Table 3", "CICIDS federation excess 0.18--0.33", [0.18, 0.33], mm([v["excess_over_seed_baseline"] for v in B100.values()]), 2)
cf = cnn["fidelity_fedavg_a100"]
chk("Table 3", "CNN federation excess 0.09--0.10 (Benign, PortScan)", [0.10, 0.09], [cf["Benign"][0], cf["PortScan"][0]], 2)
chp = cnn["holm_p"]
chk("Table 3", "CNN federation excess significant 2/6 (Benign, PortScan)", [2, 1], [sum(v < .05 for v in chp["fidelity_fedavg_a100"].values()), chp["fidelity_fedavg_a100"]["Benign"] < .05 and chp["fidelity_fedavg_a100"]["PortScan"] < .05], 0)
uf = un["B_fidelity_excess"]["100"]["per_class"]
chk("Table 3", "UNSW federation excess 0.04--0.25, 9/9 significant", [0.04, 0.25, 9], mm([v["ci"][0] for v in uf.values()]) + [sum(v["holm_p"] < .05 for v in uf.values())], 2)
chk("Table 3", "within-alpha rho CICIDS 0.23, UNSW 0.52", [0.23, 0.52], [I["within_alpha_spearman"]["rho"], un["F_drift"]["coupling"]["spearman_within_alpha"]], 2)
chk("Table 3", "FedProx dF1: CICIDS +0.15, CNN +0.08, UNSW +0.00", [0.15, 0.08, 0.00],
    [H["detection"]["0.1"]["paired_diff"][0], cnn["detection"]["fedprox_a100"][0] - cnn["detection"]["fedavg_a100"][0], un["H_fedprox"]["0.1"]["delta_macro_f1"][0]], 2)
uh = un["H_fedprox"]["0.1"]["per_class"]
chk("Table 3", "UNSW FedProx: 3/9 farther, none closer (a=0.1)", [3, 0], [sum(v["holm_p"] < .05 and v["ci"][0] > 0 for v in uh.values()), sum(v["holm_p"] < .05 and v["ci"][0] < 0 for v in uh.values())], 0)
cpx = cnn["fedprox_minus_fedavg_a100"]
chk("Table 3", "CNN FedProx: 2/6 farther, none closer (a=100)", [2, 0], [sum(chp["fedprox_minus_fedavg_a100"][k] < .05 and cpx[k][0] > 0 for k in cpx), sum(chp["fedprox_minus_fedavg_a100"][k] < .05 and cpx[k][0] < 0 for k in cpx)], 0)
chk("5.7", "CNN seed baseline 0.54--0.67", [0.54, 0.67], mm(list(cnn["seed_baseline"].values())), 2)
chk("5.7", "CNN local background (attacks) 0.42--0.70", [0.42, 0.70], mm([v[0] for k, v in cnn["background_effect_a1"].items() if k != "Benign"]), 2)
chk("5.7", "CNN drift a=0.1 0.11 +- 0.04", [0.11, 0.04], cnn["heterogeneity_drift"]["0.1"]["mean_over_classes"], 2)
ud = un["F_drift"]
chk("5.7", "UNSW drift sig. at a=5 while F1 drop not", [1, 0], [ud["5"]["class_avg_holm_p"] < .05, ud["5"]["f1_drop_holm_p"] < .05], 0)

print("=== Discussion ===")
chk("6", "DDoS a=100: EED 0.60, seed baseline 0.28, excess 0.32", [0.60, 0.28, 0.32],
    [B100["DDoS"]["EED"], sb["DDoS"]["mean"], B100["DDoS"]["excess_over_seed_baseline"]], 2)
chk("6", "DDoS F1 1.00 for FedAvg a=100 and central", [1.00, 1.00], [s7["fedavg_a100"]["per_class_f1"]["DDoS"][0], s7["central"]["per_class_f1"]["DDoS"][0]], 2)
chk("6", "client 0.46 from reference, 0.05 from class mix (a=1)", [0.46, 0.05], [D["dir_a1"]["D_client_vs_R"][0], D["dir_a1"]["D_client_vs_mixture"][0]], 2)
pl = A("privacy_leakage.json") if os.path.exists("results/analysis/privacy_leakage.json") else None
if pl:
    P17, PU = pl["cicids2017"], pl["unsw_nb15"]
    chk("6", "class-share TV error a<=1: CICIDS 0.04--0.06", [0.04, 0.06], mm([P17[a]["sigma_0.0"]["tv_est"][0] for a in ["1", "0.1"]]), 2)
    chk("6", "class-share TV error a<=1: UNSW 0.09--0.14", [0.09, 0.14], mm([PU[a]["sigma_0.0"]["tv_est"][0] for a in ["1", "0.1"]]), 2)
    chk("6", "prior-only TV a<=1 about 0.2--0.6", [0.2, 0.6], mm([d[a]["tv_prior"][0] for d in (P17, PU) for a in ["1", "0.1"]]), 1)
    chk("6", "noise 0.25/F at a=0.1: error 0.07 (CICIDS)", [0.07], [P17["0.1"]["sigma_0.25"]["tv_est"][0]], 2)
    chk("6", "noise 0.25/F at a=0.1: error about 0.2 (UNSW)", [0.2], [PU["0.1"]["sigma_0.25"]["tv_est"][0]], 1)
    chk("6", "noise 0.25/F at a=0.1: ranking shift D about 0.1", [0.1, 0.1], [P17["0.1"]["sigma_0.25"]["utility_D_noisy_vs_clean"], PU["0.1"]["sigma_0.25"]["utility_D_noisy_vs_clean"]], 1)
c6 = se["S6_cost"]
chk("6", "cost: 0.4 s per model, 0.3 s per floor, 81 numbers", [0.4, 0.3, 81], [c6["eg_seconds_R"], c6["floor_seconds_one_class"], c6["summary_floats_per_class"]], 1)

print(f"\n{n_pass} PASS, {n_fail} FAIL")
sys.exit(1 if n_fail else 0)
