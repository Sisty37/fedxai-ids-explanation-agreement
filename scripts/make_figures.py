"""Paper figures (PDF) and tables (LaTeX .tex + PDF preview) from results/analysis/*.json.
Sized for Springer LNCS (text width 12.2 cm). Fonts embedded as Type 42 (no Type 3).
Significance marks use Holm correction within each family of tests (two-sided t-test,
df = 4, reconstructed from the reported mean and 95% t-CI half-width).

Usage: python scripts/make_figures.py [--in results/analysis] [--out figures]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5, "axes.titlesize": 8,
                     "axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
                     "lines.markersize": 3.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
W = 4.8  # inches, LNCS text width
OK = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
ALPHAS = ["100", "5", "1", "0.5", "0.3", "0.1"]
CL6 = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "Bot"]
CL7 = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot"]
SHORT = {"Benign": "Benign", "DoS": "DoS", "DDoS": "DDoS", "PortScan": "PortScan",
         "BruteForce": "BruteF.", "WebAttack": "WebAtt.", "Bot": "Bot"}


DF = 4  # set from --n-seeds in main()


def p_from_ci(m, h, df=None):
    df = DF if df is None else df
    if not np.isfinite(h) or h <= 0: return 1.0
    se = h / stats.t.ppf(0.975, df)
    return float(2 * stats.t.sf(abs(m / se), df))


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); n = len(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (n - r) * p[i])); adj[i] = run
    return adj


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf")); plt.close(fig); print("wrote", name + ".pdf")


def tex_table(path, caption, label, header, rows, colspec=None):
    colspec = colspec or "l" + "c" * (len(header) - 1)
    L = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
         " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        L.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(path, "w", encoding="utf-8").write("\n".join(L).replace("±", "$\\pm$").replace("_", "\\_").replace("\\\\_", "\\_"))


def pdf_table(out, name, title, header, rows):
    rows = [r for r in rows if r != "MID"]
    fig, ax = plt.subplots(figsize=(W * 1.45, 0.3 + 0.2 * (len(rows) + 1))); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(6.5); t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0: cell.set_facecolor("#e8e8e8"); cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=7.5, loc="left"); save(fig, out, name)


def fmt(v, d=3):
    return f"{v[0]:.{d}f}±{v[1]:.{d}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default="results/analysis")
    ap.add_argument("--out", default="figures/supplement"); ap.add_argument("--n-seeds", type=int, default=5); a = ap.parse_args()
    global DF; DF = a.n_seeds - 1
    os.makedirs(a.out, exist_ok=True); tdir = os.path.join(a.out, "tables"); os.makedirs(tdir, exist_ok=True)
    J = lambda n: json.load(open(os.path.join(a.inp, n)))
    an, ex, fr, e6m, e7, dp = J("analysis.json"), J("extra.json"), J("fedprox_rq2.json"), J("e6_mechanism.json"), J("e7_ablations.json"), J("e7_dstport.json")
    H, I = fr["H_fedprox"], fr["I_coupling"]
    x = np.arange(len(ALPHAS))

    # ===== Fig. 1: detection vs explanation drift over heterogeneity and rounds =====
    fig, ax = plt.subplots(2, 2, figsize=(W, 3.5))
    c = an["C_detection"]["central"]
    ax[0, 0].axhspan(c["macro_f1"] - c["ci"], c["macro_f1"] + c["ci"], color="0.85", lw=0)
    ax[0, 0].axhline(c["macro_f1"], color="0.4", lw=0.8, ls="--", label="Centralized")
    for key, lab, col, off in [("fedavg", "FedAvg", OK[0], -0.08), ("fedprox", "FedProx ($\\mu$=0.01)", OK[1], 0.08)]:
        m = [H["detection"][al][key][0] for al in ALPHAS]; h = [H["detection"][al][key][1] for al in ALPHAS]
        ax[0, 0].errorbar(x + off, m, yerr=h, marker="o", color=col, capsize=1.5, label=lab)
    sk = an["C_detection"]["attack_skew"]
    ax[0, 0].errorbar([len(ALPHAS) + 0.3], [sk["macro_f1"]], yerr=[sk["ci"]], marker="s", color=OK[0], capsize=1.5, ls="none")
    ax[0, 0].set_xticks(list(x) + [len(ALPHAS) + 0.3], ALPHAS + ["skew"])
    ax[0, 0].set_xlabel("Dirichlet $\\alpha$ (more heterogeneous →)"); ax[0, 0].set_ylabel("Macro-F1")
    ax[0, 0].set_title("(a) Detection", loc="left"); ax[0, 0].legend(frameon=False, loc="lower left")
    on = I["onset"]
    dm = [on[al]["explanation_drift"][0] for al in ALPHAS]; dh = [on[al]["explanation_drift"][1] for al in ALPHAS]
    ddm = [on[al]["detection_drop_vs_100"][0] for al in ALPHAS]; ddh = [on[al]["detection_drop_vs_100"][1] for al in ALPHAS]
    ax[0, 1].axhline(0, color="0.5", lw=0.6)
    i03 = ALPHAS.index("0.3")
    for a_ in (ax[0, 0], ax[0, 1]):
        a_.axvspan(i03 - 0.35, i03 + 0.35, color="#FFE9A8", lw=0, zorder=0)
    ax[0, 1].text(i03, -0.065, "α=0.3", ha="center", fontsize=6, color="#8A6D00")
    ax[0, 1].errorbar(x - 0.06, dm, yerr=dh, marker="o", color=OK[2], capsize=1.5, label="Explanation drift (EED)")
    ax[0, 1].errorbar(x + 0.06, ddm, yerr=ddh, marker="^", color=OK[3], capsize=1.5, label="Macro-F1 drop")
    ax[0, 1].set_xticks(x, ALPHAS); ax[0, 1].set_xlabel("Dirichlet $\\alpha$")
    ax[0, 1].set_ylabel("Change vs. FedAvg $\\alpha$=100"); ax[0, 1].set_title("(b) Drift vs. detection drop", loc="left")
    ax[0, 1].legend(frameon=False, loc="upper left", fontsize=6)
    ax[0, 1].set_ylim(-0.085, 0.45)
    tr = an["E_trajectory"]; rounds = [int(r) for r in tr["100"]]
    TRC = ["#56B4E9", "#E69F00", "#000000"]
    for i, al in enumerate(["100", "1", "0.1"]):
        f = [tr[al][str(r)]["macro_f1"] for r in rounds]; e = [tr[al][str(r)]["fidelity_excess_mean_over_classes"] for r in rounds]
        ax[1, 0].errorbar(rounds, [v[0] for v in f], yerr=[v[1] for v in f], marker="os^"[i], color=TRC[i], capsize=1.5, label=f"$\\alpha$={al}")
        ax[1, 1].errorbar(rounds, [v[0] for v in e], yerr=[v[1] for v in e], marker="os^"[i], color=TRC[i], capsize=1.5, label=f"$\\alpha$={al}")
    ax[1, 0].set_xlabel("Round"); ax[1, 0].set_ylabel("Macro-F1"); ax[1, 0].set_title("(c) Detection over rounds", loc="left")
    ax[1, 1].set_xlabel("Round"); ax[1, 1].set_ylabel("Fidelity excess (EED)"); ax[1, 1].set_title("(d) Distance to centralized", loc="left")
    ax[1, 0].legend(frameon=False, loc="lower right"); ax[1, 1].axhline(0, color="0.5", lw=0.6)
    fig.tight_layout(h_pad=0.8, w_pad=1.0); save(fig, a.out, "fig1_detection_vs_explanation")

    # ===== Fig. 2: per-class heterogeneity drift heatmap (Holm) =====
    F = ex["F_heterogeneity_vs_alpha100"]; al5 = ALPHAS[1:]
    M = np.array([[F[al][cn][0] for al in al5] for cn in CL6]); Hh = np.array([[F[al][cn][1] for al in al5] for cn in CL6])
    padj = holm([p_from_ci(m, h) for m, h in zip(M.ravel(), Hh.ravel())]).reshape(M.shape)
    fig, axh = plt.subplots(figsize=(W * 0.62, 1.9))
    im = axh.imshow(M, cmap="YlOrRd", vmin=0, vmax=max(0.36, M.max()), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axh.text(j, i, f"{M[i, j]:.2f}{stars(padj[i, j])}", ha="center", va="center", fontsize=6,
                     color="white" if M[i, j] > 0.25 else "black")
    axh.set_xticks(range(len(al5)), al5); axh.set_yticks(range(len(CL6)), [SHORT[c] for c in CL6])
    axh.set_xlabel("Dirichlet $\\alpha$"); axh.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=axh, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=6); cb.set_label("EED vs. $\\alpha$=100", fontsize=6.5)
    save(fig, a.out, "fig2_heterogeneity_heatmap")

    # ===== Fig. 3: (a) mixture effect, (b) sources of divergence =====
    D = an["D_mixture"]; parts = ["dir_a100", "dir_a5", "dir_a1", "dir_a0.5", "dir_a0.3", "dir_a0.1", "attack_skew"]
    plab = ALPHAS + ["skew"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1, 1.35]})
    xx = np.arange(len(parts))
    for k, (key, lab, col) in enumerate([("D_client_vs_R", "vs. reference", OK[0]),
                                         ("D_client_vs_mixture", "vs. class-mix prediction", OK[4])]):
        ax[0].bar(xx + (k - 0.5) * 0.38, [D[p][key][0] for p in parts], 0.38, yerr=[D[p][key][1] for p in parts],
                  color=col, label=lab, capsize=1, error_kw={"lw": 0.6})
    ax[0].set_xticks(xx, plab, rotation=25); ax[0].set_xlabel("Partition ($\\alpha$)"); ax[0].set_ylabel("Divergence $D$ (RBO)")
    ax[0].set_title("(a) Client-level disagreement", loc="left"); ax[0].set_ylim(0, 0.8)
    ax[0].legend(frameon=False, loc="upper right", fontsize=5.6, handlelength=1.0, borderaxespad=0.1)
    G1 = ex["G_factorized"]["1"]; A = an["A_seed_baseline"]; Bf = an["B_fidelity"]["dir_a100"]
    bgx = e7["2_background"]["1"]["local_vs_shared_same_samples"]
    src = [("Data (same model)", lambda cn: G1["data"][cn][0][0], OK[5]),
           ("Retraining seed", lambda cn: A[cn]["mean"], "0.6"),
           ("FedAvg vs. central", lambda cn: Bf[cn]["excess_over_seed_baseline"], OK[0]),
           ("Heterogeneity ($\\alpha$=0.1)", lambda cn: F["0.1"][cn][0], OK[1]),
           ("SHAP background", lambda cn: bgx[cn][0], OK[3]),
           ("Local vs. global model", lambda cn: G1["model"][cn][0][0], OK[2])]
    xc = np.arange(len(CL6)); wbar = 0.8 / len(src)
    for k, (lab, fn, col) in enumerate(src):
        ax[1].bar(xc + (k - (len(src) - 1) / 2) * wbar, [fn(cn) for cn in CL6], wbar, color=col, label=lab)
    ax[1].axhline(0, color="0.5", lw=0.6); ax[1].set_xticks(xc, [SHORT[c] for c in CL6], rotation=25)
    ax[1].set_ylabel("EED (noise-floor corrected)"); ax[1].set_title("(b) Sources of explanation divergence", loc="left")
    ax[1].legend(frameon=False, fontsize=5.5, ncol=3, loc="upper center", bbox_to_anchor=(0.32, -0.27), columnspacing=0.8, handlelength=1.2)
    fig.tight_layout(w_pad=1.0); save(fig, a.out, "fig3_mixture_and_sources")

    # ===== Fig. 4: cross-dataset mechanism =====
    Am = e6m["A_attribution"]["central"]; fams = e6m["families"]; Bm = e6m["B_ablation"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.7)); xf = np.arange(len(fams))
    for k, (key, lab, col) in enumerate([("ecn_share_2017", "2017 (in-domain)", OK[0]), ("ecn_share_2018", "2018 (shifted)", OK[1])]):
        ax[0].bar(xf + (k - 0.5) * 0.38, [Am[f][key][0] for f in fams], 0.38, yerr=[Am[f][key][1] for f in fams],
                  color=col, label=lab, capsize=1, error_kw={"lw": 0.6})
    ax[0].set_xticks(xf, [SHORT[f] for f in fams]); ax[0].set_ylabel("Attribution share\nof CWR+ECE flags")
    ax[0].set_title("(a) Where explanations move", loc="left"); ax[0].set_ylim(0, 1.25); ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax[0].legend(frameon=False, loc="upper left", ncol=2, fontsize=6, handlelength=1.0)
    fl = ["Benign"] + fams
    xg = np.arange(len(fl))
    for k, (key, lab, col) in enumerate([("full", "All features", OK[0]), ("no_ecn", "Without CWR/ECE", OK[2])]):
        ax[1].bar(xg + (k - 0.5) * 0.38, [Bm[key]["f1_2018"][f][0] for f in fl], 0.38, yerr=[Bm[key]["f1_2018"][f][1] for f in fl],
                  color=col, label=lab, capsize=1, error_kw={"lw": 0.6})
    ax[1].set_xticks(xg, [SHORT[f] for f in fl]); ax[1].set_ylabel("F1 on 2018"); ax[1].set_ylim(0, 1)
    ax[1].set_title("(b) Removing the flags (centralized)", loc="left"); ax[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.0); save(fig, a.out, "fig4_cross_dataset_mechanism")

    # ===== Supplementary figures =====
    PD = H["paired_diff_prox_minus_avg"]
    M = np.array([[PD[al][cn][0] for al in ALPHAS] for cn in CL6]); Hh = np.array([[PD[al][cn][1] for al in ALPHAS] for cn in CL6])
    padj_px = holm([p_from_ci(m, h) for m, h in zip(M.ravel(), Hh.ravel())]).reshape(M.shape)
    fig, axh = plt.subplots(figsize=(W * 0.7, 1.9)); lim = np.abs(M).max()
    im = axh.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axh.text(j, i, f"{M[i, j]:+.2f}{stars(padj_px[i, j])}", ha="center", va="center", fontsize=5.8,
                     color="white" if abs(M[i, j]) > 0.6 * lim else "black")
    axh.set_xticks(range(len(ALPHAS)), ALPHAS); axh.set_yticks(range(len(CL6)), [SHORT[c] for c in CL6])
    axh.set_xlabel("Dirichlet $\\alpha$"); axh.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=axh, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=6)
    cb.set_label("Fidelity EED: FedProx − FedAvg\n(>0 = farther from centralized)", fontsize=6)
    save(fig, a.out, "figS1_fedprox_fidelity_heatmap")

    fig, axs = plt.subplots(figsize=(W * 0.6, 2.0))
    order = ALPHAS + ["skew"]; cols = {p: OK[i % len(OK)] for i, p in enumerate(order)}
    for p in order:
        rr = [r for r in I["runs"] if r["part"] == p]
        axs.scatter([r["f1_drop"] for r in rr], [r["drift"] for r in rr], s=10, color=cols[p], label=("$\\alpha$=" + p) if p != "skew" else "skew")
    ps, wa = I["pooled_spearman"], I["within_alpha_spearman"]
    axs.set_xlabel("Macro-F1 drop vs. centralized"); axs.set_ylabel("Explanation drift (EED)")
    axs.set_title(f"pooled ρ={ps['rho']:.2f} [{ps['ci'][0]:.2f},{ps['ci'][1]:.2f}]; within-α ρ={wa['rho']:.2f} [{wa['ci'][0]:.2f},{wa['ci'][1]:.2f}]", fontsize=6)
    axs.legend(frameon=False, fontsize=5.5, ncol=2); save(fig, a.out, "figS2_run_level_coupling")

    sm = dp["summary"]; cls_dp = list(sm["dst_port_share"])
    fig, axd = plt.subplots(figsize=(W * 0.55, 1.6))
    axd.bar(range(len(cls_dp)), [sm["dst_port_share"][c][0] for c in cls_dp], yerr=[sm["dst_port_share"][c][1] for c in cls_dp],
            color=OK[3], capsize=1, error_kw={"lw": 0.6})
    axd.axhline(1 / 81, color="0.4", ls="--", lw=0.7, label="uniform share (1/81)")
    axd.set_xticks(range(len(cls_dp)), [SHORT[c] for c in cls_dp]); axd.set_ylabel("Destination Port\nattribution share")
    axd.legend(frameon=False); save(fig, a.out, "figS3_dst_port_attribution")

    # ===== Tables =====
    C = an["C_detection"]
    hdr = ["Model", "Macro-F1"] + [SHORT[c] for c in CL7]
    rows = [["Centralized", fmt([C["central"]["macro_f1"], C["central"]["ci"]])] + [f"{C['central']['per_class_f1'][c][0]:.3f}" for c in CL7], "MID"]
    for p, lab in zip(parts, [f"FedAvg α={al}" for al in ALPHAS] + ["FedAvg skew"]):
        rows.append([lab, fmt([C[p]["macro_f1"], C[p]["ci"]])] + [f"{C[p]['per_class_f1'][c][0]:.3f}" for c in CL7])
    rows.append("MID")
    for al in ALPHAS:
        rows.append([f"FedProx α={al}", fmt(H["detection"][al]["fedprox"])] + ["--"] * len(CL7))
    cap = f"Detection on Test-B (mean over {a.n_seeds} seeds; macro-F1 with 95\\% t-CI; per-class F1 means)."
    tex_table(os.path.join(tdir, "tab_detection.tex"), cap, "tab:detection", hdr, rows)
    pdf_table(tdir, "tab_detection", "Table: Detection (Test-B, 5 seeds)", hdr, rows)

    hdr = ["α", "Term"] + [SHORT[c] for c in CL6]; rows = []
    for al in ["1", "0.1"]:
        for t in ["data", "model", "combined", "interaction"]:
            rows.append([al, t] + [fmt(ex["G_factorized"][al][t][cn][0], 2) for cn in CL6])
        rows.append("MID")
    rows = rows[:-1]
    tex_table(os.path.join(tdir, "tab_factorized.tex"), "Data--model factorized analysis (per-class EED, mean $\\pm$ 95\\% t-CI over 5 seeds). Not a causal decomposition.",
              "tab:factorized", hdr, rows, "ll" + "c" * len(CL6))
    pdf_table(tdir, "tab_factorized", "Table: Data-model factorized analysis", hdr, rows)

    hdr = ["α", "ΔMacro-F1"] + [SHORT[c] for c in CL6]; rows = []
    pdet = holm([p_from_ci(*H["detection"][al]["paired_diff"]) for al in ALPHAS])
    for i, al in enumerate(ALPHAS):
        d = H["detection"][al]["paired_diff"]
        rows.append([al, f"{d[0]:+.3f}{stars(pdet[i])}"] + [f"{PD[al][cn][0]:+.2f}{stars(padj_px[CL6.index(cn), i])}" for cn in CL6])
    tex_table(os.path.join(tdir, "tab_fedprox.tex"),
              "FedProx ($\\mu$=0.01) minus FedAvg, paired by seed: detection and per-class fidelity EED to the centralized reference (positive = farther). Holm-corrected: * $p<.05$, ** $p<.01$, *** $p<.001$.",
              "tab:fedprox", hdr, rows)
    pdf_table(tdir, "tab_fedprox", "Table: FedProx - FedAvg (paired; Holm stars)", hdr, rows)

    k3 = e7["3_kernelshap_vs_eg"]; bg = e7["2_background"]
    mb = e6m["B_ablation"]["paired_diff_2018_no_ecn_minus_full"]
    hdr = ["Check", "Result"]
    rows = [["RBO vs. 1−J*@10 (1,140 pairs)", f"Spearman {e7['1_metric_check']['rbo_vs_jstar10_spearman']:.2f}"],
            ["KernelSHAP vs. expected gradients", (f"overall Spearman {k3['overall_spearman']:.2f}; per class "
                                                   f"{min(v['spearman'] for v in k3['per_class'].values()):.2f}–{max(v['spearman'] for v in k3['per_class'].values()):.2f}") if isinstance(k3, dict) else k3],
            ["Local vs. shared SHAP background (α=1)", f"EED Benign {bg['1']['local_vs_shared_same_samples']['Benign'][0]:.2f}; attacks "
             f"{min(v[0] for k, v in bg['1']['local_vs_shared_same_samples'].items() if k != 'Benign'):.2f}–{max(v[0] for k, v in bg['1']['local_vs_shared_same_samples'].items() if k != 'Benign'):.2f}"],
            ["Destination Port included", f"macro-F1 {fmt(sm['macro_f1_main'])} → {fmt(sm['macro_f1_with_port'])}; rank 1 for every class"],
            ["PortScan rows after dedupe", f"1,669 (no port) vs. {dp['class_counts_with_port']['PortScan']:,} (with port)"],
            ["2018 F1 without CWR/ECE (paired)", f"DDoS {mb['DDoS'][0]:+.3f}±{mb['DDoS'][1]:.3f}; Bot {mb['Bot'][0]:+.3f}±{mb['Bot'][1]:.3f}"]]
    tex_table(os.path.join(tdir, "tab_robustness.tex"), "Robustness and sensitivity checks.", "tab:robustness", hdr, rows, "ll")
    pdf_table(tdir, "tab_robustness", "Table: Robustness checks", hdr, rows)

    # Holm summary for the text
    sig = {"heterogeneity_cells_sig": int((padj < 0.05).sum()), "heterogeneity_cells_total": int(padj.size),
           "alpha0.1_all_classes_sig": bool((padj[:, -1] < 0.05).all()),
           "fedprox_fidelity_cells_sig": int((padj_px < 0.05).sum()), "fedprox_cells_total": int(padj_px.size)}
    json.dump(sig, open(os.path.join(a.out, "holm_summary.json"), "w", encoding="utf-8"), indent=1); print("Holm:", sig)


if __name__ == "__main__":
    main()
