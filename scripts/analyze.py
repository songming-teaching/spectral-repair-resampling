"""Analysis + figures for EXP-1/2/3 and Table I.  Usage: python analyze.py <expdir>"""
import os, sys, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srcore import *

E = sys.argv[1]
D, F, RS = [os.path.join(E, x) for x in ("data", "figures", "results")]
os.makedirs(F, exist_ok=True)
PR = json.load(open(os.path.join(RS, "prereg_constants.json")))

# a recorded design decision presentation deltas: the measured quantities (fixed-bandwidth Stieltjes deviation + h/2
# sensitivity, the six-group SE table, the bootstrap 95% CI on the log-log slope, Table I ESS/sec).
# They are produced by checks/figure_additions.py, which only MEASURES already-produced data.  Read it
# if present and render it into the panels; a missing file is reported, never silently skipped -- an
# annotation that quietly vanishes leaves the figure looking complete while the numbers are absent.
# The stage/flat layout puts it under the stage root, the repo layout under results/window/.
FA = None
for _p in (os.path.join(E, "figure_additions.json"), os.path.join(E, "..", "figure_additions.json"),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "window",
                        "figure_additions.json")):
    if os.path.exists(_p):
        FA = json.load(open(_p)); FA_SRC = _p; break
if FA is None:
    print("WARNING: figure_additions.json not found -- decision_13 annotations omitted "
          "(run checks/figure_additions.py)")
summary = {}
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
INV_ST = {"none": ("k", "no correction"), "oracle": ("C2", "oracle T"),
          "cond": ("C0", "conditional plug-in"), "realized": ("C3", "realized counts")}

def med_iqr(v):
    v = np.asarray(v, float); return float(np.median(v)), float(np.quantile(v, .25)), float(np.quantile(v, .75))

# =========================== EXP-1 ===========================
R1 = json.load(open(os.path.join(D, "exp1", "exp1_rows.json")))
A1 = np.load(os.path.join(D, "exp1", "exp1_arrays.npz"))
c, tauH, gH = 0.75, np.array([0.5, 2.5]), np.array([0.7, 0.3])
a1, m2H = 1.1, 2.05
grp = collections.defaultdict(list)
for r in R1: grp[(r["scheme"], r["alpha"], r["N"], r["inverter"])].append(r)

# Fig 1(a)
# a recorded design decision asks for four presentation deltas on this figure.  Their numbers are too long to sit
# inside the data panels without covering the data, so the three panels occupy the top row and a
# dedicated strip below carries them: a reader sees the plot and its numbers, neither on top of the
# other.
fig = plt.figure(figsize=(10.5, 4.1))
_gs = fig.add_gridspec(2, 3, height_ratios=[3.1, 1.0], hspace=0.42)
axs = [fig.add_subplot(_gs[0, i]) for i in range(3)]
ax_txt = fig.add_subplot(_gs[1, :]); ax_txt.axis("off")
ax = axs[0]
_gof = (FA or {}).get("fig1a_goodness_of_fit", {})
for scheme, col in [("multi", "C0"), ("res", "C3")]:
    eig = np.concatenate([A1[k] for k in A1.files if k.startswith("N2000_a1.0_") and k.endswith(f"{scheme}_eigs")])
    pos, p0 = split_atom(eig)
    nu = trim_pmf(nu_oracle_multi_gamma(1.0) if scheme == "multi" else nu_oracle_res_gamma(1.0))
    xs = np.linspace(0.01, np.quantile(pos, 0.97), 600)
    p0t = zero_atom(nu[0], c)
    zz = xs + 1e-4j
    m, *_ = stieltjes(zz, tauH, gH, np.arange(len(nu), dtype=float), nu, c)
    dens = (m + p0t / zz).imag / np.pi
    ax.hist(pos, bins=120, range=(0, xs[-1]), weights=np.full(len(pos), 1.0 / len(eig)) / (xs[-1] / 120),
            histtype="stepfilled", alpha=0.3, color=col)
    ax.plot(xs, dens, color=col, lw=1.2, label=f"{scheme}: theory")
    ax.plot([0], [p0], "o", color=col, ms=5); ax.plot([0], [p0t], "x", color="k", ms=6)
    summary.setdefault("exp1_fig1a", {})[scheme] = dict(atom_emp=float(p0), atom_theory=float(p0t))
# a recorded design decision: quantitative goodness of fit instead of "the curves look close".  The resolved-interior
# L1 is the number that is a statement about the theory; the full-support L1/sup are dominated by the
# near-zero edge and histogram resolution there (see checks/figure_additions.py's note), so the
# interior value is the headline and the full-support value is carried alongside it.
_gl = "   ".join(
    f"{s}:  L1$_{{\\rm int}}$={_gof[s]['L1_resolved_interior']:.3f}"
    f" (full {_gof[s]['L1_density_deviation']:.3f}),  sup$_{{\\rm int}}$={_gof[s]['sup_resolved_interior']:.3f},"
    f"  h/2 max rel. change $\\leq${_gof[s]['h_half_max_rel_change_interior']:.1e}"
    for s in ("multi", "res") if s in _gof)
if _gl:
    ax_txt.text(0.0, 0.86, "(a) fit quality — " + _gl, fontsize=5.6, color="k", va="top", ha="left")
ax.set_title("(a) resampled ESD, N=2000, α=1 (pooled 50 reps)\nzero atom: o = empirical, x = theory", fontsize=8); ax.set_ylim(0, 0.9)
ax.set_xlabel("eigenvalue"); ax.legend(fontsize=6.5)
# Fig 1(b)
ax = axs[1]
fit = {}
for scheme, mk in [("multi", "o"), ("res", "^")]:
    for al, col in [(10.0, "C2"), (1.0, "C0"), (0.3, "C1")]:
        rows = [r for r in R1 if r["scheme"] == scheme and r["alpha"] == al and r["inverter"] == "none" and r["N"] == 2000]
        x = np.array([(1 + r["r2N"]) if scheme == "multi" else (r["r2N"] + r["EU_N"]) for r in rows])
        y = np.array([r["m2_emp"] for r in rows])
        ax.plot(x, y, mk, ms=2.5, color=col, alpha=0.6, label=f"{scheme} α={al:g}")
        fit[f"{scheme}_a{al}"] = dict(mean_resid=float(np.mean(y - (m2H + c * a1 ** 2 * x))),
                                      se=float(np.std(y - (m2H + c * a1 ** 2 * x), ddof=1) / np.sqrt(len(y))),
                                      mean_resid_given_counts=float(np.mean(
                                          y - np.array([m2H * (1 + r["b2N"] / r["N"]) + c * a1 ** 2 * r["b2N"] for r in rows]))))
xx = np.linspace(1.5, 9, 10); ax.plot(xx, m2H + c * a1 ** 2 * xx, "k-", lw=1, label="theory")
# a recorded design decision: write the six-group SE table out instead of saying "about 2 SE".  Each group shows
# mean residual (+- its own SE) and the resulting t = mean/SE, so a reader checks significance
# directly rather than trusting a phrase.  |t| <= 2.16 for all six means every group is inside ~2 SE.
_se = (FA or {}).get("fig1b_se_table", {})
if _se:
    _rows = "   ".join(f"{k.replace('_a',' α=')}: {v['mean_resid']:+.4f}±{v['se']:.4f} (t={v['t_like']:+.2f})"
                       for k, v in sorted(_se.items()))
    ax_txt.text(0.0, 0.52, "(b) six-group mean residual ± SE  —  " + _rows, fontsize=5.6, color="k",
                va="top", ha="left")
ax.set_xlabel("predicted count 2nd moment  m₂(ν̂)"); ax.set_ylabel("m̂₂ = tr(S²)/d")
ax.set_title("(b) second-moment certificate, N=2000", fontsize=8); ax.legend(fontsize=5.5, ncol=2)
summary["exp1_fig1b_residuals"] = fit
# Fig 1(c)
ax = axs[2]
Ns = [500, 1000, 2000]; tab1c = {}
for scheme, ls in [("multi", "-"), ("res", "--")]:
    for inv, (col, lab) in INV_ST.items():
        st = [med_iqr([r["W1"] for r in grp[(scheme, 1.0, N, inv)]]) for N in Ns]
        tab1c[f"{scheme}_{inv}"] = st
        if inv == "none": continue
        ax.errorbar(Ns, [s[0] for s in st], yerr=[[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]],
                    color=col, ls=ls, marker="o", ms=3, capsize=2, label=f"{lab} ({scheme})" if scheme == "multi" else None)
ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off(); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
ax.set_xlabel("N"); ax.set_ylabel("W₁(Ĥ, H_N)")
lim = {(l["scheme"], l["alpha"]): l for l in PR["no_correction_limit"]}
ax.axhline(lim[("multi", 1.0)]["W1"], color="k", ls=":", lw=1, label="no correction: prereg. limit")
# a recorded design decision: 95% CI on the log-log slope, not just the point estimate.  The slope is the rate the
# main text quotes; a bare "-0.49" invites the reader to treat it as exact.  Bootstrap over reps.
_sl = (FA or {}).get("fig1c_slope_ci", {})
_lab = {"oracle": "oracle", "cond": "cond", "realized": "realized"}
_sl_rows = []
for _inv in ("oracle", "cond", "realized"):
    _a = _sl.get(f"multi|a=1.0|{_inv}"); _b = _sl.get(f"res|a=1.0|{_inv}")
    if _a and _b:
        _sl_rows.append(f"{_lab[_inv]}:  multi {_a['slope']:+.2f} [{_a['ci95'][0]:+.2f},{_a['ci95'][1]:+.2f}]"
                        f"   res {_b['slope']:+.2f} [{_b['ci95'][0]:+.2f},{_b['ci95'][1]:+.2f}]")
if _sl_rows:
    ax_txt.text(0.0, 0.20, "(c) log-log slope of median W₁ vs N, α=1, 95% CI by bootstrap over reps  —  "
                + "     ".join(_sl_rows), fontsize=5.6, color="k", va="top", ha="left")
ax.set_title("(c) inversion error, α=1 (solid multi, dashed res)", fontsize=8); ax.legend(fontsize=6)
fig.savefig(os.path.join(F, "fig1_exp1.pdf")); fig.savefig(os.path.join(F, "fig1_exp1.png")); plt.close(fig)

full = {}
for (scheme, al, N, inv), rows in sorted(grp.items()):
    full[f"{scheme}|a={al}|N={N}|{inv}"] = dict(W1=med_iqr([r["W1"] for r in rows]), W2=med_iqr([r["W2"] for r in rows]),
        eps=med_iqr([r.get("eps_model", np.nan) for r in rows]) if inv != "none" else None)
summary["exp1_table"] = full
nc = {}
for (scheme, al), l in lim.items():
    obs = [r["W1"] for r in grp[(scheme, al, 2000, "none")]]
    nc[f"{scheme}_a{al}"] = dict(prereg=l["W1"], observed_median_N2000=float(np.median(obs)))
summary["exp1_no_correction_vs_prereg"] = nc
# rates (slope of log median W1 vs log N)
rates = {}
for scheme in ["multi", "res"]:
    for al in [10.0, 1.0, 0.3]:
        for inv in ["oracle", "cond", "realized"]:
            y = [np.median([r["W1"] for r in grp[(scheme, al, N, inv)]]) for N in Ns]
            rates[f"{scheme}|a={al}|{inv}"] = float(np.polyfit(np.log(Ns), np.log(y), 1)[0])
summary["exp1_log_slopes_W1_vs_N"] = rates
# order test cond vs realized (paired, N=2000)
order = {}
for scheme in ["multi", "res"]:
    for al in [10.0, 1.0, 0.3]:
        a_ = {r["rep"]: r["W1"] for r in grp[(scheme, al, 2000, "cond")]}
        b_ = {r["rep"]: r["W1"] for r in grp[(scheme, al, 2000, "realized")]}
        o_ = {r["rep"]: r["W1"] for r in grp[(scheme, al, 2000, "oracle")]}
        ks = sorted(a_)
        order[f"{scheme}_a{al}"] = dict(frac_realized_better=float(np.mean([b_[k] < a_[k] for k in ks])),
                                        p_wilcoxon_cond_vs_realized=float(wilcoxon([a_[k] for k in ks], [b_[k] for k in ks]).pvalue),
                                        frac_cond_better_than_oracle=float(np.mean([a_[k] < o_[k] for k in ks])))
summary["exp1_order_N2000"] = order

# =========================== EXP-2 ===========================
R2 = json.load(open(os.path.join(D, "exp2", "exp2_rows.json")))
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
Ra = [r for r in R2 if r["part"] == "a"]
order_s = ["S1 multi->multi", "S2 res->res", "S3 multi->res (wrong mechanism)",
           "S4c permuted control q=10", "S4 endogenous q=10", "S4c permuted control q=750", "S4 endogenous q=750",
           "S5 correlated particles rho=0.5", "S5 correlated particles rho=0.9"]
data = [[r["eps_model"] for r in Ra if r["setting"] == s] for s in order_s]
ax = axs[0]
ax.boxplot(data, vert=True, showfliers=False)
ax.set_xticks(range(1, len(order_s) + 1))
ax.set_xticklabels(["S1", "S2", "S3\nwrong\nmech.", "S4c\nq=10", "S4\nq=10", "S4c\nq=750", "S4\nq=750", "S5\nρ=.5", "S5\nρ=.9"], fontsize=6.5)
ax.set_yscale("log"); ax.set_ylabel("ε̂_model"); ax.set_title("(a) mismatch certificate, N=2000, α=1", fontsize=8)
base = np.array(data[0])
tests = {}
for s, dd in zip(order_s, data):
    tests[s] = dict(median=float(np.median(dd)), q25=float(np.quantile(dd, .25)), q75=float(np.quantile(dd, .75)),
                    p_vs_S1_greater=float(mannwhitneyu(dd, base, alternative="greater").pvalue),
                    W1_median=float(np.median([r["W1"] for r in Ra if r["setting"] == s])))
for q in [10, 750]:
    x = [r["eps_model"] for r in Ra if r["setting"] == f"S4 endogenous q={q}"]
    y = [r["eps_model"] for r in Ra if r["setting"] == f"S4c permuted control q={q}"]
    tests[f"S4 vs S4c q={q} (two-sided)"] = float(mannwhitneyu(x, y).pvalue)
summary["exp2a"] = tests
Rb = [r for r in R2 if r["part"] == "b"]
ax = axs[1]
cm = collections.defaultdict(list)
for r in Rb: cm[(r["c"], r["alpha"])].append(r)
X_ = np.array([v[0]["c_minus_ps"] for v in cm.values()])
Y_ = np.array([np.median([r["W1"] for r in v]) for v in cm.values()])
K_ = np.array([np.median([r["kappa_W1"] for r in v]) for v in cm.values()])
AT = np.array([max(0, 1 - v[0]["p_s"] / v[0]["c"]) for v in cm.values()])
sc = ax.scatter(AT, Y_, c=np.log10(K_), cmap="viridis", s=40, edgecolor="k", lw=0.4)
plt.colorbar(sc, ax=ax, label="log₁₀ κ̂_inv^(W₁)")
ax.set_xlabel("zero-atom mass  1 − p_s/c  (clipped at 0)"); ax.set_ylabel("median W₁(Ĥ, δ₁)")
ax.set_title("(b) identifiability, H=δ₁, N=1000\ncolour = log₁₀ κ̂_inv (v6: zero atom is the driver)", fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig2_exp2.pdf")); fig.savefig(os.path.join(F, "fig2_exp2.png")); plt.close(fig)
summary["exp2b"] = dict(
    spearman_kappa_vs_abs_c_minus_ps=[float(x) for x in spearmanr(np.abs(X_), K_)],
    spearman_W1_vs_zero_atom=[float(x) for x in spearmanr(AT, Y_)],
    spearman_kappa_vs_zero_atom=[float(x) for x in spearmanr(AT, K_)],
    spearman_W1_vs_kappa=[float(x) for x in spearmanr(K_, Y_)],
    cells={f"c={k[0]},a={k[1]}": dict(c_minus_ps=v[0]["c_minus_ps"], zero_atom=max(0, 1 - v[0]["p_s"] / v[0]["c"]),
           W1=med_iqr([r["W1"] for r in v]), kappa=med_iqr([r["kappa_W1"] for r in v])) for k, v in sorted(cm.items())})
fig, ax = plt.subplots(figsize=(3.6, 3))
ax.scatter(AT, K_, c=[v[0]["c"] for v in cm.values()], cmap="plasma", edgecolor="k", lw=0.4)
ax.set_xlabel("zero-atom mass 1 − p_s/c (clipped)"); ax.set_ylabel("median κ̂_inv^(W₁)"); ax.set_yscale("log")
ax.set_title("S5 supplement: κ̂ is governed by lost rank", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(F, "figS_kappa_vs_atom.png")); plt.close(fig)

# =========================== EXP-3 ===========================
p3 = os.path.join(D, "exp3", "exp3_rows.json")
if os.path.exists(p3):
    R3 = json.load(open(p3)); K = 32; a2 = 0.64
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in R3: by[r["method"]][r["rep"]][r["k"]] = r
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
    ks = np.arange(K + 1)
    def traj(m, key="eta_W1"):
        return np.array([[by[m][rep][k][key] for k in ks] for rep in sorted(by[m])])
    ax = axs[0]
    for m, col, lab in [("none", "k", "no correction"), ("cond", "C0", "conditional plug-in"),
                        ("oracle", "C2", "oracle correction"), ("sup", "C1", "plug-in + supervisor")]:
        T = traj(m); md = np.median(T, 0)
        ax.plot(ks, md, color=col, lw=1.2, label=lab)
        ax.fill_between(ks, np.quantile(T, .25, 0), np.quantile(T, .75, 0), color=col, alpha=0.15)
    for s in [8, 16, 24]: ax.axvline(s, color="r", ls=":", lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("step k"); ax.set_ylabel("η_k = W₁(H_k⁺, H_k^⋆)")
    ax.set_title("(a) stable scalar cycle (red: weight shocks α=0.3)", fontsize=8); ax.legend(fontsize=6)
    ax = axs[1]
    for m, col, lab in [("forced_end", "C3", "forced abstention: end {27..32}"),
                        ("forced_uniform", "C4", "forced abstention: uniform {5,..,30}"), ("cond", "C0", "no abstention")]:
        T = traj(m); ax.plot(ks, np.median(T, 0), color=col, lw=1.2, label=lab)
        ax.fill_between(ks, np.quantile(T, .25, 0), np.quantile(T, .75, 0), color=col, alpha=0.15)
    ax.set_yscale("log"); ax.set_xlabel("step k"); ax.set_title("(b) same A_K = 6, different timing", fontsize=8); ax.legend(fontsize=6)
    ax = axs[2]
    pts = []
    for m in ["forced_end", "forced_uniform", "sup", "cond"]:
        for rep in by[m]:
            ab = [k for k in ks if by[m][rep][k]["abstain"]]
            for k in ks[1:]:
                Lam = sum(a2 ** (k - j) for j in ab if j <= k)
                pts.append((Lam, by[m][rep][k]["eta_W1"], m))
    P = np.array([(p[0], p[1]) for p in pts])
    ax.plot(P[:, 0], P[:, 1], ".", ms=2, alpha=0.3, label="(k, rep) pairs")
    bbar = float(np.quantile([r["b_W1"] for r in R3 if r["method"] == "cond"], 0.95))
    dbar = float(np.quantile([r["delta_fb_W1"] for r in R3 if r["method"] == "cond"], 0.95))
    LL = np.linspace(0, P[:, 0].max(), 50)
    ax.plot(LL, bbar / (1 - a2) + dbar * LL, "r-", lw=1, label="bound  b̄/(1−ā²) + δ̄Λ  (95% b̄, δ̄)")
    ax.legend(fontsize=6)
    summary.setdefault("exp3_bound", {}).update(bbar95=bbar, dbar95=dbar,
        frac_below_bound=float(np.mean(P[:, 1] <= bbar / (1 - a2) + dbar * P[:, 0])))
    ax.set_xlabel("geometric abstention load Λ_k"); ax.set_ylabel("η_k (W₁)"); ax.set_yscale("log")
    ax.set_title("(c) abstention cost is driven by Λ_k", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig3_exp3.pdf")); fig.savefig(os.path.join(F, "fig3_exp3.png")); plt.close(fig)
    e_end = traj("forced_end")[:, K]; e_uni = traj("forced_uniform")[:, K]
    s3 = dict(eta_K_forced_end=med_iqr(e_end), eta_K_forced_uniform=med_iqr(e_uni),
              p_end_greater_uniform=float(wilcoxon(e_end, e_uni, alternative="greater").pvalue),
              spearman_Lambda_vs_eta=[float(x) for x in spearmanr(P[:, 0], P[:, 1])])
    for m in by:
        T = traj(m); s3[f"{m}_eta"] = dict(k0=med_iqr(T[:, 0]), k8=med_iqr(T[:, 8]), k9=med_iqr(T[:, 9]),
                                           k16=med_iqr(T[:, 16]), k32=med_iqr(T[:, 32]), max_median=float(np.max(np.median(T, 0))),
                                           abstentions_total=int(sum(by[m][rep][k]["abstain"] for rep in by[m] for k in ks)))
    # recursion bound check eta_k <= a^2 eta_{k-1} + B_k  (B_k = b_k or delta_fb,k)
    chk = []
    for m in ["cond", "oracle", "sup", "forced_end", "forced_uniform"]:
        for rep in by[m]:
            for k in ks[1:]:
                r, rp = by[m][rep][k], by[m][rep][k - 1]
                B = r["delta_fb_W1"] if r["abstain"] else r["b_W1"]
                chk.append(r["eta_W1"] <= a2 * rp["eta_W1"] + B + 1e-3)
    s3["recursion_bound_holds_frac"] = float(np.mean(chk))
    # per-step budget scale
    s3["b_W1_median_corrected"] = float(np.median([r["b_W1"] for r in R3 if r["method"] == "cond"]))
    s3["delta_fb_W1_median"] = float(np.median([r["delta_fb_W1"] for r in R3 if r["method"] == "cond"]))
    summary["exp3"] = s3

# =========================== Table I ===========================
pt = os.path.join(D, "tableI", "tableI.json")
if os.path.exists(pt):
    T1 = json.load(open(pt)); summary["tableI"] = T1

json.dump(summary, open(os.path.join(RS, "summary.json"), "w"), indent=1, default=float)
print(json.dumps({k: summary[k] for k in summary if k not in ("exp1_table", "tableI")}, indent=1, default=float)[:6000])
