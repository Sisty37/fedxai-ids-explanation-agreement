"""Print the key numbers reported in the paper next to the values in the result files,
so they can be checked by eye. Usage: python scripts/check_numbers.py
"""
import json, os
import numpy as np

A = lambda n: json.load(open(os.path.join("results", "analysis", n), encoding="utf-8"))
an, ex, fr, e6, em, e7, dp = (A("analysis.json"), A("extra.json"), A("fedprox_rq2.json"), A("e6.json"),
                               A("e6_mechanism.json"), A("e7_ablations.json"), A("e7_dstport.json"))
cnn = A("cnn_validation.json") if os.path.exists("results/analysis/cnn_validation.json") else None
CL = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "Bot"]
rows = []
def add(sec, paper, value): rows.append((sec, paper, value))
rng = lambda v: f"{min(v):.2f}--{max(v):.2f}"

D = an["D_mixture"]
add("5.1 / Abstract", "attack mix explains 82--93%", rng([d["share_explained_by_mix"] * 100 for d in D.values()]) + " %")
add("5.1", "client vs reference D = 0.33--0.54", rng([d["D_client_vs_R"][0] for d in D.values()]))
add("5.1", "client vs mixture 0.04--0.06", rng([d["D_client_vs_mixture"][0] for d in D.values()]))
bg = e7["2_background"]["1"]["local_vs_shared_same_samples"]
add("5.1 / Abstract", "background: Benign 0.07, attacks 0.30--0.64", f"Benign {bg['Benign'][0]:.2f}, attacks " + rng([v[0] for k, v in bg.items() if k != "Benign"]))
sb = an["A_seed_baseline"]
add("5.2", "seed baseline Benign 0.02, attacks 0.13--0.28", f"Benign {sb['Benign']['mean']:.2f}, attacks " + rng([v["mean"] for k, v in sb.items() if k != "Benign"]))
G = ex["G_factorized"]
add("5.2", "model term 0.45--0.75 (a=1), 0.55--0.71 (a=0.1)", rng([G['1']['model'][c][0][0] for c in CL]) + " / " + rng([G['0.1']['model'][c][0][0] for c in CL]))
add("5.2", "FedAvg a=100 beyond seed 0.18--0.33", rng([v["excess_over_seed_baseline"] for v in an["B_fidelity"]["dir_a100"].values()]))
tr = an["E_trajectory"]["100"]
add("5.2", "fidelity excess 0.49 (r1) -> ~0.24 (r30+)", f"r1 {tr['1']['fidelity_excess_mean_over_classes'][0]:.2f}, r30 {tr['30']['fidelity_excess_mean_over_classes'][0]:.2f}, r50 {tr['50']['fidelity_excess_mean_over_classes'][0]:.2f}")
C = an["C_detection"]
add("5.2", "macro-F1 FedAvg a=100 0.933 vs central 0.966", f"{C['dir_a100']['macro_f1']:.3f} vs {C['central']['macro_f1']:.3f}")
on = fr["I_coupling"]["onset"]
add("5.3", "drift 0.002/0.039/0.073/0.111/0.276 (a=5,1,0.5,0.3,0.1)", " / ".join(f"{on[a]['explanation_drift'][0]:.3f}" for a in ["5", "1", "0.5", "0.3", "0.1"]))
add("5.3", "macro-F1 a=0.1 0.648 +- 0.095", f"{C['dir_a0.1']['macro_f1']:.3f} +- {C['dir_a0.1']['ci']:.3f}")
I = fr["I_coupling"]
add("5.3", "pooled rho 0.62 [0.43,0.75]; within 0.23 [-0.06,0.48]",
    f"{I['pooled_spearman']['rho']:.2f} {np.round(I['pooled_spearman']['ci'], 2).tolist()}; {I['within_alpha_spearman']['rho']:.2f} {np.round(I['within_alpha_spearman']['ci'], 2).tolist()}")
H = fr["H_fedprox"]
add("5.4", "FedProx a=0.1: 0.648 -> 0.800 (+0.152)", f"{H['detection']['0.1']['fedavg'][0]:.3f} -> {H['detection']['0.1']['fedprox'][0]:.3f} ({H['detection']['0.1']['paired_diff'][0]:+.3f})")
mu = H["mu_sweep_alpha0.1"]
add("5.4", "mu sweep fidelity 0.32/0.45/0.51, F1 0.76/0.80/0.74",
    " / ".join(f"{np.mean([mu[m]['fidelity'][c][0] for c in CL]):.2f}" for m in ["0.001", "0.01", "0.1"]) + ", F1 " + " / ".join(f"{mu[m]['macro_f1'][0]:.2f}" for m in ["0.001", "0.01", "0.1"]))
cells = [c for v in e6["models"].values() for c in v["cells"]]
add("5.5", "240 cells all significant, EED 0.23--0.84", f"{sum(c['significant'] for c in cells)}/{len(cells)}, EED {min(c['dE'] for c in cells):.2f}--{max(c['dE'] for c in cells):.2f}")
Am = em["A_attribution"]
add("5.5", "CWR+ECE share 88% (central), 80% (FedAvg)", f"{Am['central']['DDoS']['ecn_share_2018'][0]:.3f}/{Am['central']['Bot']['ecn_share_2018'][0]:.3f}, {Am['fedavg_a100']['DDoS']['ecn_share_2018'][0]:.3f}/{Am['fedavg_a100']['Bot']['ecn_share_2018'][0]:.3f}")
B = em["B_ablation"]
add("5.5", "no-ECN: 2017 0.969 vs 0.966; DDoS 0.22+-0.08, Bot 0.29+-0.28",
    f"{B['no_ecn']['testb_macro_f1'][0]:.3f} vs {B['full']['testb_macro_f1'][0]:.3f}; DDoS {B['paired_diff_2018_no_ecn_minus_full']['DDoS'][0]:.2f}+-{B['paired_diff_2018_no_ecn_minus_full']['DDoS'][1]:.2f}, Bot {B['paired_diff_2018_no_ecn_minus_full']['Bot'][0]:.2f}+-{B['paired_diff_2018_no_ecn_minus_full']['Bot'][1]:.2f}")
k3 = e7["3_kernelshap_vs_eg"]
add("5.6", "RBO vs J* 0.76 (4,680 pairs); KernelSHAP 0.91, per class 0.36--0.78",
    f"{e7['1_metric_check']['rbo_vs_jstar10_spearman']:.2f} ({e7['1_metric_check']['n_pairs']}); {k3['overall_spearman']:.2f}, " + rng([v['spearman'] for v in k3['per_class'].values()]))
add("5.6", "Dst port: PortScan 1,669 -> 90,613; F1 0.960 -> 0.966",
    f"{dp['class_counts_with_port']['PortScan']}; {dp['summary']['macro_f1_main'][0]:.3f} -> {dp['summary']['macro_f1_with_port'][0]:.3f}")
if cnn:
    add("5.6 CNN", "F1 0.961 / 0.537; mix 75--80%; seed 0.52--0.66",
        f"{cnn['detection']['central'][0]:.3f} / {cnn['detection']['fedavg_a0.1'][0]:.3f}; " + rng([d['share_explained_by_mix'] * 100 for d in cnn['class_mix'].values()]) + "%; " + rng(list(cnn['seed_baseline'].values())))
    add("5.6 CNN", "drift a=0.1 0.108+-0.029; background 0.43--0.71",
        f"{cnn['heterogeneity_drift']['0.1']['mean_over_classes'][0]:.3f}+-{cnn['heterogeneity_drift']['0.1']['mean_over_classes'][1]:.3f}; " + rng([v[0] for k, v in cnn['background_effect_a1'].items() if k != 'Benign']))

w = max(len(r[1]) for r in rows)
print(f"{'Section':15s} {'Paper says':{w}s} | File says")
print("-" * (w + 45))
for sec, paper, val in rows:
    print(f"{sec:15s} {paper:{w}s} | {val}")
