r"""s6_burnin.py -- S6 MCMC diagnostics: the burn-in certificate (a recorded design decision, S6 row).

a recorded design decision asks for "burn-in diagnostics (drift ratio, skewness, BURN/tau; the m(t) negative result
was deleted under D18, so rewrite the design-certificate sentence)".  What is plotted here is the
DESIGN CERTIFICATE, not a convergence proof: at BURN = 2000 the Robbins-Monro step-size controller has
settled (drift ratio small) and the remaining dependence is carried explicitly by tau, which the
production run reports per chain.

Panels:
  (a) the RM step-size trajectory rm_traj, log scale, with the burn-in cutoff -- the thing BURN is
      supposed to be past;
  (b) the tail drift ratio rm_tail_drift_ratio per (regime, preconditioner): the certificate's own
      number, with the pre-registered threshold drawn;
  (c) BURN/tau per chain, against the threshold -- this is the "is the burn-in window long enough in
      units of the chain's own correlation time" question, and the answer is recorded per chain
      rather than averaged, because averaging is exactly what hides a short chain.

D18 NOTE: the m(t) curve was a NEGATIVE result and was deleted under D18; it is not reproduced here.

Usage:  python s6_burnin.py <stage>
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = sys.argv[1]
D, F, RS = [os.path.join(E, x) for x in ("data", "figures", "results")]
os.makedirs(F, exist_ok=True)
R = json.load(open(os.path.join(D, "burnin_diag", "burnin_diag.json")))

PRE = ["I", "S_r^c + eps I", "lift(H_hat) + eps I", "oracle Sigma + eps I"]
NL = {"I": "$I$", "S_r^c + eps I": "$S_r^c+\\varepsilon I$",
      "lift(H_hat) + eps I": "lift", "oracle Sigma + eps I": "oracle"}
REG = ["A_strong", "G21"]
COLS = {"I": "C0", "S_r^c + eps I": "C1", "lift(H_hat) + eps I": "C2", "oracle Sigma + eps I": "C3"}
# The TWO pre-stated reading rules, taken verbatim from the producer
# (archive/code_2026-09-27/burnin_diag.py, "PRE-STATED READING RULES (fixed before looking at the
# numbers)").  They are NOT invented here: a threshold that a figure asserts must be the one the
# experiment was read against, or the figure is grading itself.
#   tail-drift ratio < 0.2  => Robbins-Monro has effectively settled
#   BURN / tau        >= 5  => burn-in budget is adequate (5-10 mixing times is the common rule);
#                              < 5 would mean burn-in may be too short for slowly mixing chains
DRIFT_THRESH = 0.2
TAU_FLOOR = 5.0

INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))

# ---- (a) the RM step-size trajectory ------------------------------------------------------------
ax = axs[0]
shown = 0
for r in R:
    if r["regime"] != "A_strong" or not isinstance(r["rm_traj"], list):
        continue
    ax.plot(np.abs(r["rm_traj"]), color=COLS[r["preconditioner"]], lw=0.7, alpha=0.8,
            label=NL[r["preconditioner"]] if shown < 4 else None)
    shown += 1
    if shown >= 4:
        break
ax.axvline(R[0]["burn"], color=INK, ls="--", lw=1)
ax.text(R[0]["burn"] * 1.1, ax.get_ylim()[0] * 1.5, f"BURN={R[0]['burn']}", fontsize=6.5, color=INK)
ax.set_yscale("log"); ax.set_xlabel("MH step"); ax.set_ylabel(r"$|\log \varepsilon|$ (RM controller)")
ax.set_title("(a) Robbins--Monro settles before BURN", fontsize=8, loc="left")
ax.legend(fontsize=6, frameon=False, loc="upper right")

# ---- (b) tail drift ratio ------------------------------------------------------------------------
ax = axs[1]
xs, ys, cs = [], [], []
for i, reg in enumerate(REG):
    for p in PRE:
        v = [r["rm_tail_drift_ratio"] for r in R if r["regime"] == reg and r["preconditioner"] == p]
        for j, vv in enumerate(v):
            xs.append(i + (j - 0.5) * 0.18); ys.append(vv); cs.append(COLS[p])
ax.scatter(xs, ys, c=cs, s=26, edgecolor="k", lw=0.3, zorder=3)
ax.axhline(DRIFT_THRESH, color=INK, ls="--", lw=1)
ax.text(-0.42, DRIFT_THRESH * 1.12, f"pre-stated settled $<{DRIFT_THRESH:g}$", fontsize=6, ha="left", color=INK)
ax.set_yscale("log"); ax.set_xticks(range(len(REG)))
ax.set_xticklabels(["A (high $r$, low $c$)", "G21 (high $r$, high $c$)"], fontsize=6.5)
ax.set_ylabel("tail drift ratio")
_n_settled = sum(1 for v in ys if v < DRIFT_THRESH)
ax.set_title(f"(b) {_n_settled}/{len(ys)} controllers settled at BURN", fontsize=8, loc="left")

# ---- (c) BURN/tau per chain ----------------------------------------------------------------------
ax = axs[2]
vals = []
for i, reg in enumerate(REG):
    for p in PRE:
        for r in R:
            if r["regime"] == reg and r["preconditioner"] == p and isinstance(r["burnin_over_tau"], list):
                for t in r["burnin_over_tau"]:
                    vals.append((i, p, float(t)))
pos = {}
n = 0
for p in PRE:
    pos[p] = n; n += 1
for i, p, t in vals:
    ax.plot(i + (pos[p] - 1.5) * 0.19, t, "o", ms=3, color=COLS[p], alpha=0.75,
            markeredgecolor="k", markeredgewidth=0.2)
ax.axhline(TAU_FLOOR, color=INK, ls="--", lw=1)
ax.text(1.45, TAU_FLOOR * 1.08, f"pre-stated floor {TAU_FLOOR:g} mixing times", fontsize=6, ha="right", color=INK)
ax.set_yscale("log"); ax.set_xticks(range(len(REG)))
ax.set_xticklabels(["A", "G21"], fontsize=7)
ax.set_ylabel(r"BURN $/\ \tau$  (per chain)")
ax.set_title("(c) burn-in window in units of $\\tau$, per chain", fontsize=8, loc="left")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=COLS[p], label=NL[p],
                          markeredgecolor="k", markeredgewidth=0.2) for p in PRE],
          fontsize=6, frameon=False, loc="lower left", ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(F, "figS6_burnin.pdf")); fig.savefig(os.path.join(F, "figS6_burnin.png"))
plt.close(fig)

# ---- the certificate, as numbers -----------------------------------------------------------------
cert = {}
for reg in REG:
    for p in PRE:
        sub = [r for r in R if r["regime"] == reg and r["preconditioner"] == p]
        drifts = [r["rm_tail_drift_ratio"] for r in sub]
        bt = [t for r in sub if isinstance(r["burnin_over_tau"], list) for t in r["burnin_over_tau"]]
        cert[f"{reg}|{p}"] = dict(
            n_seeds=len(sub), burn=sub[0]["burn"] if sub else None,
            drift_ratio_min=min(drifts), drift_ratio_max=max(drifts),
            drift_settled=bool(max(drifts) < DRIFT_THRESH),
            burn_over_tau_min=min(bt), burn_over_tau_max=max(bt),
            burn_over_tau_median=float(np.median(bt)),
            chains_below_floor=int(sum(1 for t in bt if t < TAU_FLOOR)), chains_total=len(bt),
            rm_converged=int(sum(1 for r in sub if r["rm_converged"])), n=int(len(sub)))
json.dump(dict(drift_threshold=DRIFT_THRESH, tau_floor=TAU_FLOOR, burn=2000, per_cell=cert,
               rules_source=("archive/code_2026-09-27/burnin_diag.py PRE-STATED READING RULES "
                             "(drift ratio < 0.2 settled; BURN/tau >= 5 adequate; < 5 too short)"),
               note=("Design certificate, not a convergence proof.  D18: the m(t) curve was a negative "
                     "result and was deleted; not reproduced here.")),
          open(os.path.join(RS, "s6_burnin_summary.json"), "w"), indent=1)

print("=== S6 burn-in certificate (pre-stated rules: drift < 0.2 settled; BURN/tau >= 5 adequate) ===")
print(f"  {'cell':34s} {'drift max':>9s} {'settled':>8s} {'B/tau med':>10s} {'B/tau min':>10s} "
      f"{'below 5':>9s} {'rm_conv':>8s}")
for k, v in cert.items():
    print(f"  {k:34s} {v['drift_ratio_max']:9.4f} {str(v['drift_settled']):>8s} "
          f"{v['burn_over_tau_median']:10.2f} {v['burn_over_tau_min']:10.2f} "
          f"{v['chains_below_floor']:>4d}/{v['chains_total']:<4d} "
          f"{v['rm_converged']:>4d}/{v['n']:<3d}")
