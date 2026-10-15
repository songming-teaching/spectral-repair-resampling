r"""s5_phasediagram.py -- S5: the two-layer phase diagram + the cross-experiment consistency note.

a recorded design decision's S5 row asks for the phase diagram with a TWO-LAYER narrative (map level x pipeline
level) and, in place of the ROC, the abstention audit.

WHY THERE IS NO ROC (a recorded design decision point 3).  The ROC's positive class is empty: on EXP-3 the correct
truth (same-state comparison eta_cond(k) > eta_fb_same_state(k)) has ZERO positives in 660 steps --
correction beats the fallback at every state.  A ROC with no positives is undefined, so drawing one
would be a figure asserting something it cannot measure.  What IS drawn instead is the cross-experiment
consistency note: the distribution of the supervisor's own decision ratio Bhat/delta_fb on EXP-3,
against the boundary the S7 scan says has to be crossed for clause (b) to fire.  That is the honest
version of "where would this rule switch on", and it is measurable.

Panels:
  (a) MAP level  -- EXP-2b: median kappa_inv^(W1) over (c, alpha).  How well the inversion can
      identify, as a function of the two design knobs;
  (b) PIPELINE level -- S7: clause (b) activation fraction over (c, alpha) at the deployed bandwidth.
      The same axes, a different question: not "can we identify" but "does the supervisor SAY so";
  (c) the audit -- the Bhatt/delta_fb ratio distribution, with the activation boundary.

Usage:  python s5_phasediagram.py <stage>
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import collections

E = sys.argv[1]
D, F, RS = [os.path.join(E, x) for x in ("data", "figures", "results")]
os.makedirs(F, exist_ok=True)

R2 = [r for r in json.load(open(os.path.join(D, "exp2", "exp2_rows.json"))) if r["part"] == "b"]
S7 = json.load(open(os.path.join(D, "s7", "s7_activation_M15_refit.json")))
R3 = [r for r in json.load(open(os.path.join(D, "exp3", "exp3_rows.json"))) if r["method"] == "sup"]

DEPLOYED = "0.03"
# THE TWO LAYERS COME FROM DIFFERENT EXPERIMENTS AND DO NOT SHARE A GRID.  EXP-2b runs
# c in {0.5,0.75,1.0,1.25} x alpha in {0.3,0.5,1.0,3.0,10.0}; S7 runs c in {0.15,...,1.25} x
# alpha in {0.3,1.0,3.0,10.0}.  Plotting them on a common axis requires the INTERSECTION, and the
# cells outside it must be named rather than silently dropped (a figure that quietly intersects two
# grids shows a smaller experiment than the one that was run).
CS = sorted({r["c"] for r in R2} & {r["c"] for r in S7})
ALS = sorted({r["alpha"] for r in R2} & {r["alpha"] for r in S7})
EXCLUDED = dict(exp2b_only=sorted({(r["c"], r["alpha"]) for r in R2} - set(zip(
                            [r["c"] for r in S7], [r["alpha"] for r in S7])) ),
                s7_only_c=sorted({r["c"] for r in S7} - set(CS)),
                s7_only_alpha=sorted({r["alpha"] for r in S7} - set(ALS)))

# ---- (a) map level: median kappa_inv ------------------------------------------------------------
K = np.full((len(ALS), len(CS)), np.nan)
for i, a in enumerate(ALS):
    for j, c in enumerate(CS):
        v = [r["kappa_W1"] for r in R2 if r["c"] == c and r["alpha"] == a]
        if v:
            K[i, j] = np.median(v)

# ---- (b) pipeline level: activation fraction -----------------------------------------------------
A = np.full((len(ALS), len(CS)), np.nan)
for i, a in enumerate(ALS):
    for j, c in enumerate(CS):
        v = [r["by_h"][DEPLOYED]["activated"] for r in S7 if r["c"] == c and r["alpha"] == a]
        if v:
            A[i, j] = np.mean(v)

INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))

def grid_axes(ax, title):
    ax.set_xticks(range(len(CS))); ax.set_xticklabels([f"{c:g}" for c in CS], fontsize=7)
    ax.set_yticks(range(len(ALS))); ax.set_yticklabels([f"{a:g}" for a in ALS], fontsize=7)
    ax.set_xlabel("$c = d/N$"); ax.set_ylabel("weight-law $\\alpha$")
    ax.set_title(title, fontsize=8, loc="left")

ax = axs[0]
im = ax.imshow(K, origin="lower", aspect="auto", cmap="viridis",
               norm=matplotlib.colors.LogNorm(vmin=np.nanmin(K), vmax=np.nanmax(K)))
for i in range(len(ALS)):
    for j in range(len(CS)):
        if np.isfinite(K[i, j]):
            ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
plt.colorbar(im, ax=ax, label="median $\\hat\\kappa_{\\rm inv}^{(W_1)}$")
grid_axes(ax, "(a) MAP level: how identifiable the inverse problem is")

ax = axs[1]
im = ax.imshow(A, origin="lower", aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
for i in range(len(ALS)):
    for j in range(len(CS)):
        if np.isfinite(A[i, j]):
            ax.text(j, i, f"{A[i, j]:.0%}", ha="center", va="center", fontsize=6)
# mark the EXP-3 deployment point
dj = CS.index(0.75); di = ALS.index(3.0)
ax.add_patch(plt.Rectangle((dj - .5, di - .5), 1, 1, fill=False, edgecolor="k", lw=1.8))
ax.text(dj + .55, di, "EXP-3\n(deployed)", fontsize=5.5, va="center", color="k")
plt.colorbar(im, ax=ax, label="clause (b) activation fraction")
grid_axes(ax, "(b) PIPELINE level: where the supervisor says so")

# ---- (c) the audit in place of the ROC -----------------------------------------------------------
ax = axs[2]
ratio = np.array([r["B_hat"] / r["delta_fb_W1"] for r in R3 if r["delta_fb_W1"] > 0])
ax.hist(ratio, bins=40, color="#2a78d6", alpha=0.85, edgecolor="k", lw=0.3)
ax.axvline(1.0, color=INK, ls="--", lw=1.2)
ax.text(1.02, ax.get_ylim()[1] * 0.95, "clause (b)\nboundary\n$\\hat{B}/\\hat\\delta_{fb}=1$",
        fontsize=6, va="top", color=INK)
ax.set_xlabel("$\\hat{B}\\,/\\,\\hat\\delta_{fb}$  (supervisor's decision ratio)")
ax.set_ylabel("count (660 steps)")
med = float(np.median(ratio)); mx = float(np.max(ratio))
ax.set_title(f"(c) audit: ratio median {med:.3f}, max {mx:.3f} $<1$ -- never fires", fontsize=8, loc="left")
ax.text(0.02, 0.97, "empty positive class:\nno ROC is drawable\n(decision_12)", transform=ax.transAxes,
        fontsize=6, va="top", color=INK2)

fig.tight_layout()
fig.savefig(os.path.join(F, "figS5_phasediagram.pdf"))
fig.savefig(os.path.join(F, "figS5_phasediagram.png"))
plt.close(fig)

summary = dict(
    deployed_h=DEPLOYED, grid=dict(c=CS, alpha=ALS), grid_excluded=EXCLUDED,
    map_level_kappa={f"c={c},a={a}": (None if not np.isfinite(K[i, j]) else float(K[i, j]))
                     for i, a in enumerate(ALS) for j, c in enumerate(CS)},
    pipeline_level_activation={f"c={c},a={a}": (None if not np.isfinite(A[i, j]) else float(A[i, j]))
                               for i, a in enumerate(ALS) for j, c in enumerate(CS)},
    audit=dict(n=len(ratio), median_ratio=med, max_ratio=mx,
               fraction_above_boundary=float(np.mean(ratio > 1.0)),
               exp3_deployment=dict(c=0.75, alpha=3.0,
                                    activation=float(A[ALS.index(3.0), CS.index(0.75)]))),
    no_roc=("decision_12: the same-state truth has 0 positives in 660 steps, so the ROC's positive "
            "class is empty and no ROC is drawable; the cross-experiment ratio audit stands in."),
    two_layers=("(a) is the MAP-layer question (can the inversion identify?), (b) is the PIPELINE-layer "
                "question (does the supervisor activate?); the same (c, alpha) axes carry both."))
json.dump(summary, open(os.path.join(RS, "s5_phasediagram_summary.json"), "w"), indent=1)

print("=== S5 two-layer phase diagram ===")
print(f"  map level  kappa_inv: min {np.nanmin(K):.3f}  max {np.nanmax(K):.3f}")
print(f"  pipeline activation : overall {np.nanmean(A):.3f}  at EXP-3 "
      f"{A[ALS.index(3.0), CS.index(0.75)]:.3f}")
print(f"  audit ratio: n={len(ratio)} median {med:.4f} max {mx:.4f} "
      f"frac>1 = {np.mean(ratio > 1.0):.3f}")
