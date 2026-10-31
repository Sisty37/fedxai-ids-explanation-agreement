"""Paper figures for the camera-ready version (run after the analysis scripts):
  Fig. 2: CICIDS2017 vs. UNSW-NB15, sources of divergence and drift vs. detection
  Fig. 3: CICIDS2017 detection vs. heterogeneity and distance to centralized over rounds
Inputs: results/analysis/{analysis,e7_ablations,fedprox_rq2,sensitivity}.json, results/unsw/unsw_analysis.json
Usage:  python scripts/make_paper_figures.py      Output: figures/paper/fig2_sources_two_datasets.pdf, figures/paper/fig3_detection_and_rounds.pdf
"""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
os.makedirs("figures/paper", exist_ok=True)
J = lambda p: json.load(open(p))

# ---------------- Fig. 2 ----------------
U = J("results/unsw/unsw_analysis.json")
S = J("results/analysis/sensitivity.json")
A = J("results/analysis/analysis.json")
E7 = J("results/analysis/e7_ablations.json")
I = J("results/analysis/fedprox_rq2.json")["I_coupling"]["runs"]
C17 = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "Bot"]
UC = [c for c in U["classes_analyzed"] if U["BG_background_alpha1"]["local_vs_shared_same_flows"][c][1] >= 30]
cic = [[A["D_mixture"]["dir_a1"]["per_class_client_EED"][c]["mean"] for c in C17],
       [E7["2_background"]["1"]["local_vs_shared_same_samples"][c][0] for c in C17],
       [A["A_seed_baseline"][c]["mean"] for c in C17],
       [S["S1_rbo_p"]["0.9"]["federation_excess"][c]["ci"][0] for c in C17],
       [S["S1_rbo_p"]["0.9"]["drift_alpha0.1"][c]["ci"][0] for c in C17]]
uns = [[U["D_mixture"]["1"]["per_class_client_EED"][c]["mean"] for c in UC],
       [U["BG_background_alpha1"]["local_vs_shared_same_flows"][c][0] for c in UC],
       [U["A_seed_baseline"][c]["mean"] for c in UC],
       [U["B_fidelity_excess"]["100"]["per_class"][c]["ci"][0] for c in UC],
       [U["F_drift"]["0.1"]["per_class"][c][0] for c in UC]]
labels = ["Client vs.\nreference", "Local\nbackground", "Retraining\n(seeds)", "Federation\n($\\alpha$=100)", "Heterog.\n($\\alpha$=0.1)"]
col = {"CICIDS2017": "#0072B2", "UNSW-NB15": "#D55E00"}
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.8, "axes.labelsize": 6.5, "legend.fontsize": 5.6, "xtick.labelsize": 5.8, "ytick.labelsize": 6, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(4.8, 1.72), gridspec_kw={"width_ratios": [1.55, 1]})
rng = np.random.default_rng(0)
x = np.arange(len(labels)); w = 0.36
for k, (name, data) in enumerate([("CICIDS2017", cic), ("UNSW-NB15", uns)]):
    off = (k - 0.5) * w
    ax[0].bar(x + off, [np.mean(v) for v in data], w * 0.92, color=col[name], alpha=0.35, label=name, edgecolor=col[name], lw=0.6)
    for i, v in enumerate(data):
        ax[0].scatter(x[i] + off + rng.uniform(-w * 0.25, w * 0.25, len(v)), v, s=4, color=col[name], zorder=3, lw=0)
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=5.6)
ax[0].set_ylabel("EED")
ax[0].set_title("(a) Sources of divergence", loc="left")
ax[0].legend(frameon=False, loc="upper right", handlelength=1)
# (b)
def tci(v):
    v = np.asarray(v, float); return v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
al = ["5", "1", "0.5", "0.3", "0.1"]; xb = np.arange(len(al))
base = {r["seed"]: r["f1_drop"] for r in I if r["part"] == "100"}
cd = [tci([r["drift"] for r in I if r["part"] == a]) for a in al]
cf = [tci([r["f1_drop"] - base[r["seed"]] for r in I if r["part"] == a]) for a in al]
ud = [U["F_drift"][a]["class_avg"] for a in al]; uf = [U["F_drift"][a]["f1_drop"] for a in al]
for name, d, f, o in [("CICIDS2017", cd, cf, -0.06), ("UNSW-NB15", ud, uf, 0.06)]:
    ax[1].errorbar(xb + o, [v[0] for v in d], yerr=[v[1] for v in d], color=col[name], marker="o", ms=2.5, lw=0.9, capsize=1.5, label=f"{name} drift")
    ax[1].errorbar(xb + o, [v[0] for v in f], yerr=[v[1] for v in f], color=col[name], marker="s", ms=2.2, lw=0.9, ls="--", capsize=1.5, mfc="white", label=f"{name} F1 drop")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xticks(xb); ax[1].set_xticklabels(al); ax[1].set_xlabel("Dirichlet $\\alpha$")
ax[1].set_ylabel("Change vs. $\\alpha$=100")
ax[1].set_title("(b) Drift vs. detection", loc="left")
ax[1].legend(frameon=False, loc="upper left", fontsize=5.2, handlelength=1.6, labelspacing=0.25)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig("figures/paper/fig2_sources_two_datasets.pdf"); plt.close(fig)
print("wrote figures/paper/fig2_sources_two_datasets.pdf")

# ---------------- Fig. 3 ----------------
plt.rcdefaults()
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.2, "lines.markersize": 3.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
OK = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]; ALPHAS = ["100", "5", "1", "0.5", "0.3", "0.1"]
an = J("results/analysis/analysis.json"); H = J("results/analysis/fedprox_rq2.json")["H_fedprox"]
x = np.arange(len(ALPHAS))
fig, ax = plt.subplots(1, 2, figsize=(4.8, 1.55))
c = an["C_detection"]["central"]
ax[0].axhspan(c["macro_f1"] - c["ci"], c["macro_f1"] + c["ci"], color="0.85", lw=0)
ax[0].axhline(c["macro_f1"], color="0.4", lw=0.8, ls="--", label="Centralized")
for key, lab, col, off in [("fedavg", "FedAvg", OK[0], -0.08), ("fedprox", "FedProx ($\\mu$=0.01)", OK[1], 0.08)]:
    ax[0].errorbar(x + off, [H["detection"][a][key][0] for a in ALPHAS], yerr=[H["detection"][a][key][1] for a in ALPHAS], marker="o", color=col, capsize=1.5, label=lab)
sk = an["C_detection"]["attack_skew"]
ax[0].errorbar([len(ALPHAS) + 0.3], [sk["macro_f1"]], yerr=[sk["ci"]], marker="s", color=OK[0], capsize=1.5, ls="none")
ax[0].set_xticks(list(x) + [len(ALPHAS) + 0.3], ALPHAS + ["skew"]); ax[0].set_xlabel("Dirichlet $\\alpha$ (more heterogeneous →)")
ax[0].set_ylabel("Macro-F1"); ax[0].set_title("(a) Detection", loc="left"); ax[0].legend(frameon=False, loc="lower left")
tr = an["E_trajectory"]; rounds = [int(r) for r in tr["100"]]; TRC = ["#56B4E9", "#E69F00", "#000000"]
for i, al in enumerate(["100", "1", "0.1"]):
    e = [tr[al][str(r)]["fidelity_excess_mean_over_classes"] for r in rounds]
    ax[1].errorbar(rounds, [v[0] for v in e], yerr=[v[1] for v in e], marker="os^"[i], color=TRC[i], capsize=1.5, label=f"$\\alpha$={al}")
ax[1].axhline(0, color="0.5", lw=0.6); ax[1].set_xlabel("Round"); ax[1].set_ylabel("Fidelity excess (EED)")
ax[1].set_title("(b) Distance to centralized", loc="left"); ax[1].legend(frameon=False, loc="lower left", ncol=3, columnspacing=0.8, handletextpad=0.3); ax[1].set_ylim(-0.24, 0.58)
fig.tight_layout(w_pad=1.0); fig.savefig("figures/paper/fig3_detection_and_rounds.pdf"); print("wrote figures/paper/fig3_detection_and_rounds.pdf")
