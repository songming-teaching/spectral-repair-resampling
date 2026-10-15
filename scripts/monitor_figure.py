r"""monitor_figure.py -- the S-Monitor figure (a recorded design decision section 1 / M42).

A passive monitor on a PLAIN bootstrap PF: no correction chain, no inversion, just the spectral
instrument attached -- at every step it records ESS (the weight side), the zero-atom mass and
d_eff (the last two from the count-rank identity, which costs no eigendecomposition).  The weight
law manufactures a degeneracy at the EXP-3 shocks (k in {8,16,24}, alpha -> 0.3).

M42 PREREGISTERED the prediction that the spectral traces alert while ESS stays SILENT.  The run
REFUTED it: at the shocks ESS falls 3.23x while the zero atom rises 2.29x, and the two move
together (corr -0.987, no lead/lag).  Under the preregistered judgment rule selling point 2
downgrades to "the spectral diagnostic COMPLEMENTS ESS rather than replacing it".

This figure reports that outcome AS MEASURED -- it does not pick the outcome.  Panel (a) the three
trajectories normalised to their steady mean; (b) the ESS / zero-atom plane; (c) the shock response
against the preregistered "ESS is silent" line.

Usage:  python monitor_figure.py <stage>
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = sys.argv[1]
D, F, RS = [os.path.join(E, x) for x in ("data", "figures", "results")]
os.makedirs(F, exist_ok=True)
N = 1000
SHOCKS = (8, 16, 24)
QUIET = 1.5                       # the preregistered "ESS is silent" threshold, from checks/monitor_demo.py

R = [r for r in json.load(open(os.path.join(D, "monitor", "exp3_rows.json"))) if r["method"] == "none"]
for r in R:
    r["ESS"] = N / r["r2N"]
ks = sorted({r["k"] for r in R})
ess = np.array([r["ESS"] for r in R])
za = np.array([r["p0_count_rank"] for r in R])
de = np.array([r["d_eff"] for r in R])
shock = np.array([r["k"] in SHOCKS for r in R])
steady = ~shock
byk = lambda k: [i for i, r in enumerate(R) if r["k"] == k]

INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
C_ESS, C_ZA, C_DE = "#2a78d6", "#eb6834", "#1baf7a"

fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))

# ---- (a) the three trajectories, each normalised to its own steady mean -------------------------
ax = axs[0]
for lab, v, col in [("ESS", ess, C_ESS), ("zero atom", za, C_ZA), (r"$d_{\rm eff}$", de, C_DE)]:
    md = np.array([np.median(v[byk(k)]) for k in ks])
    q1 = np.array([np.quantile(v[byk(k)], .25) for k in ks])
    q3 = np.array([np.quantile(v[byk(k)], .75) for k in ks])
    base = md[[i for i, k in enumerate(ks) if k not in SHOCKS]].mean()
    ax.plot(ks, md / base, color=col, lw=1.3, label=lab)
    ax.fill_between(ks, q1 / base, q3 / base, color=col, alpha=0.16, lw=0)
for s in SHOCKS:
    ax.axvline(s, color=INK2, ls=":", lw=0.8)
ax.axhline(1.0, color=INK2, lw=0.7, ls="--")
ax.text(SHOCKS[0] + 0.4, ax.get_ylim()[1] * 0.98, "weight shocks", fontsize=6, color=INK2, va="top")
ax.set_ylim(0.2, 2.6)
ax.set_xlabel("step $k$"); ax.set_ylabel("value / steady mean")
ax.set_title("(a) all three move together", fontsize=8, loc="left")
ax.legend(fontsize=6.5, frameon=False, loc="lower right")

# ---- (b) the ESS / zero-atom plane --------------------------------------------------------------
ax = axs[1]
c_all = float(np.corrcoef(ess, za)[0, 1])
ax.plot(ess[steady], za[steady], ".", ms=4, alpha=0.45, color=C_ESS, label="steady")
ax.plot(ess[shock], za[shock], "o", ms=4.5, alpha=0.75, color=C_ZA, label="at a shock")
ax.set_xlabel("ESS"); ax.set_ylabel("zero-atom mass")
ax.set_title(f"(b) corr = {c_all:+.3f}: one curve, no separation", fontsize=8, loc="left")
ax.legend(fontsize=6.5, frameon=False, loc="upper right")

# ---- (c) shock response vs the preregistered line ------------------------------------------------
ax = axs[2]
r_ess = float(ess[steady].mean() / ess[shock].mean())
r_za = float(za[shock].mean() / za[steady].mean())
r_de = float(de[steady].mean() / de[shock].mean())
ax.bar([0, 1, 2], [r_ess, r_za, r_de], width=0.6, color=[C_ESS, C_ZA, C_DE], edgecolor=INK, lw=0.5)
ax.axhline(QUIET, color=INK, ls="--", lw=1)
ax.text(2.42, QUIET + 0.07, f"preregistered\n'ESS is silent'\n(ratio < {QUIET:g})",
        fontsize=6, ha="right", va="bottom", color=INK)
for x, v in zip([0, 1, 2], [r_ess, r_za, r_de]):
    ax.text(x, v + 0.07, f"{v:.2f}x", ha="center", fontsize=6.5, color=INK)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["ESS\n(falls)", "zero atom\n(rises)", r"$d_{\rm eff}$" + "\n(falls)"], fontsize=6.5)
ax.set_ylim(0, 4.2)
ax.set_ylabel("shock / steady ratio")
ax.set_title("(c) ESS is NOT quiet $\\Rightarrow$ prediction fails", fontsize=8, loc="left")

fig.tight_layout()
fig.savefig(os.path.join(F, "figS_monitor.pdf"))
fig.savefig(os.path.join(F, "figS_monitor.png"))
plt.close(fig)

lead = {}
for lag in (-1, 0, 1):
    xs, ys = [], []
    for r in R:
        m = [q for q in R if q["rep"] == r["rep"] and q["k"] == r["k"] + lag]
        if m:
            xs.append(r["ESS"]); ys.append(m[0]["p0_count_rank"])
    lead[str(lag)] = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else None

summary = dict(
    shocks=list(SHOCKS), threshold_ESS_is_silent=QUIET,
    steady=dict(ess=float(ess[steady].mean()), zero_atom=float(za[steady].mean()), d_eff=float(de[steady].mean())),
    shock=dict(ess=float(ess[shock].mean()), zero_atom=float(za[shock].mean()), d_eff=float(de[shock].mean())),
    ratio=dict(ess=r_ess, zero_atom=r_za, d_eff=r_de),
    corr_ess_zero_atom=c_all, lead_lag=lead,
    verdict=("PREDICTION FAILS: ESS falls 3.23x at the shocks, exceeding the preregistered 'ESS is silent' "
             "threshold of 1.5.  ESS and the spectral traces move together (corr -0.987, no lead/lag), so "
             "selling point 2 downgrades to 'the spectral diagnostic COMPLEMENTS ESS rather than replacing it'."),
    note=("M42 preregistered the prediction that the spectral traces alert while ESS is silent; this "
          "figure reports what was measured, per the preregistered judgment rule."))
json.dump(summary, open(os.path.join(RS, "monitor_summary.json"), "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("ratio", "corr_ess_zero_atom", "lead_lag")}, indent=1))
