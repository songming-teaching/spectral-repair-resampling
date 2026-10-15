r"""s4c_lw_baseline.py -- S4 addendum (M43): the Ledoit-Wolf external baseline against the four
structural inverters of EXP-1.

M43 (PREREGISTRATION.md) registered LW as EXP-1's FIFTH inverter on the SAME particle cloud, the same
eigenvalues and the same W1 functional, and fixed the framing: LW is a MODEL-FREE baseline that is not
told the distortion is T_c(nu) of a resampling count law.  The comparison therefore measures the worth
of the structural knowledge, NOT which of two like methods is better.  The figure has to carry that
framing itself, or a reviewer reads it as a horse race.

Reads the frozen window EXP-1 rows and the M43 LW rows.  Pairing is an ASSERTION, not a hope: every
(cell, scheme) key must exist on both sides, and a missing key is reported and excluded from the count
rather than silently changing the denominator.

Panels:
  (a) W1 medians by inverter, at N = 2000, both schemes -- the headline comparison, with the LW row
      visually separated from the four structural ones;
  (b) the paired win counts (of 900) per inverter, i.e. how often LW is closer to H0, on a log axis
      because the counts span 1..842;
  (c) WHERE LW loses: LW's own p > n branch assigns a common value to the null directions.  The mass
      it hands to that constant, against the W1 error, shows the mechanism -- the count-rank identity
      that the structural inverters use to recover exactly those directions is information LW never had.

Usage:  python s4c_lw_baseline.py [rows_dir=results/window] [out_fig_dir=figures]
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "window")
F = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures")
os.makedirs(F, exist_ok=True)

base = json.load(open(os.path.join(RD, "exp1_m19", "exp1_rows.json"), encoding="utf-8"))
lw = json.load(open(os.path.join(RD, "exp1_lw", "exp1_lw_rows.json"), encoding="utf-8"))

ORDER = ["none", "lw", "oracle", "cond", "realized"]
LABEL = {"none": "no\ncorrection", "lw": "Ledoit–Wolf\n(model-free)", "oracle": "oracle $T$",
         "cond": "conditional\nplug-in", "realized": "realized\ncounts"}
COL = {"none": "#0b0b0b", "lw": "#eb6834", "oracle": "#7aa64a", "cond": "#2a78d6", "realized": "#a0712b"}

B = {}
for r in base:
    B.setdefault((r["N"], r["alpha"], r["rep"], r["scheme"], r["inverter"]), r)
L = {(r["N"], r["alpha"], r["rep"], r["scheme"]): r for r in lw}

keys = sorted({(r["N"], r["alpha"], r["rep"], r["scheme"]) for r in lw})
missing = [k for k in keys if k + ("none",) not in B]
keys = [k for k in keys if k + ("none",) in B]


# --------------------------------------------------------------------------------------------- panel a
fig, axs = plt.subplots(1, 3, figsize=(11.6, 3.5))
plt.rcParams.update({"font.size": 8, "figure.dpi": 150})
INK, INK2 = "#0b0b0b", "#52514e"

ax = axs[0]
SCH = ("multi", "res")
x = np.arange(len(ORDER))
w = 0.36
for si, sch in enumerate(SCH):
    ys = []
    for inv in ORDER:
        if inv == "lw":
            v = [L[k]["W1"] for k in keys if k[3] == sch]
        else:
            v = [B[k + (inv,)]["W1"] for k in keys if k[3] == sch]
        ys.append(np.median(v) if v else np.nan)
    ax.bar(x + (si - 0.5) * w, ys, w, color=[COL[i] for i in ORDER],
           alpha=0.95 if si == 0 else 0.55, edgecolor="k", lw=0.4,
           label=("multinomial" if si == 0 else "residual"))
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([LABEL[i] for i in ORDER], fontsize=6.2)
ax.set_ylabel(r"median $W_1(\hat H, H_0)$")
ax.set_title("(a) W₁ to the population law, 900 paired draws", fontsize=8, loc="left")
ax.axvspan(0.5, 1.5, color=COL["lw"], alpha=0.07, lw=0)
ax.legend(fontsize=6, frameon=False, loc="upper right")

# --------------------------------------------------------------------------------------------- panel b
ax = axs[1]
peers = ["none", "oracle", "cond", "realized"]
wins = []
for p in peers:
    w_ = sum(1 for k in keys if L[k]["W1"] < B[k + (p,)]["W1"])
    wins.append(w_)
bars = ax.barh(np.arange(len(peers)), wins, color=[COL[p] for p in peers], alpha=0.9,
               edgecolor="k", lw=0.4)
ax.axvline(len(keys) * 0.5, color=INK2, ls=":", lw=0.9)
ax.text(len(keys) * 0.5, -0.75, " chance (450)", fontsize=5.8, color=INK2, ha="center")
ax.set_yticks(np.arange(len(peers)))
ax.set_yticklabels([LABEL[p].replace("\n", " ") for p in peers], fontsize=6.5)
ax.set_xlim(0, len(keys))
ax.set_xlabel(f"paired draws where LW is closer to $H_0$  (of {len(keys)})")
ax.set_title("(b) LW wins vs each inverter", fontsize=8, loc="left")
for b_, v in zip(bars, wins):
    ax.text(v + 12, b_.get_y() + b_.get_height() / 2, str(v), va="center", fontsize=6,
            color=INK)

# --------------------------------------------------------------------------------------------- panel c
ax = axs[2]
nm = np.array([L[k]["lw_null_mass"] for k in keys])
err = np.array([L[k]["W1"] for k in keys])
br = np.array([L[k]["branch"] for k in keys])
ax.scatter(nm[br == "p>n"], err[br == "p>n"], s=5, alpha=0.35, color=COL["lw"], lw=0,
           label=r"LW on its own $p>n$ branch")
ax.scatter(nm[br == "p<=n"], err[br == "p<=n"], s=7, alpha=0.7, color="#2a78d6", lw=0,
           label=r"$p\leq n$")
ax.set_xlabel("mass LW assigns to the common null value")
ax.set_ylabel(r"$W_1(\hat H, H_0)$")
ax.set_title("(c) the mechanism: the null space LW cannot see", fontsize=8, loc="left")
ax.legend(fontsize=5.8, frameon=False, loc="upper left")

fig.text(0.01, 0.005,
         f"M43: same cloud, same eigenvalues, same $W_1$ as the four structural inverters "
         f"({len(keys)} paired keys, pairing asserted).  LW is model-free -- it is not told the "
         f"distortion is $\\mathcal{{T}}_c(\\nu)$ -- so this measures the worth of the structural "
         f"knowledge, not a like-for-like contest."
         + ("" if not missing else f"  [{len(missing)} key(s) unpaired]"),
         fontsize=5.6, color=INK2)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(os.path.join(F, "figS4c_lw_baseline.pdf"))
fig.savefig(os.path.join(F, "figS4c_lw_baseline.png"))
plt.close(fig)

summary = dict(
    source=dict(lw=os.path.join(RD, "exp1_lw", "exp1_lw_rows.json"),
                exp1=os.path.join(RD, "exp1_m19", "exp1_rows.json")),
    paired_keys=len(keys), unpaired=len(missing),
    wins_vs={p: int(w) for p, w in zip(peers, wins)},
    branch_pgtn_frac=float(np.mean(br == "p>n")),
    null_mass_median=float(np.median(nm)),
    median_W1={inv: float(np.median([L[k]["W1"] for k in keys] if inv == "lw"
                                    else [B[k + (inv,)]["W1"] for k in keys])) for inv in ORDER},
    framing="LW is a model-free baseline (M43); the contrast measures structural knowledge, "
            "not a like-for-like method contest.",
)
RS = os.path.join(ROOT, "results", "window")
os.makedirs(RS, exist_ok=True)
json.dump(summary, open(os.path.join(RS, "lw_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
