r"""s4_sensitivity.py -- S4: inversion sensitivity.

a recorded design decision's S4 row lists five items:
  (1) the gamma-scale incident            -> checks/gamma_incident.json (data package)
  (2) the M14/M15 two-version numbers     -> data/m17/s7_full_compare.json         [present]
  (3) delta_fb vs N  (M24)                -> NOT COMPUTED                          [ABSENT]
  (4) the two gamma rules side by side    -> data/gamma_rule_probe.json            [present]
  (5) the seven solver unit tests         -> results/solver_unit_tests.json        [present]

ITEM (3) IS NOT DRAWN, AND SAYS SO.  M24 (PREREGISTRATION.md) ruled that the main text says "no
closed-form correction" and S4 carries a MEASURED delta_fb-vs-N curve.  That sweep was never run --
there is no producer and no output file in the tree (checked).  Rather than interpolate a curve from
unrelated runs, panel (c) is left as an explicit placeholder naming the missing measurement.  A figure
that invents the curve it was supposed to measure is worse than a figure with a hole in it.

Panels: (a) the gamma incident -- same eigenvalues, two gamma rules, against the regime's own spectrum;
(b) the two rules' relative W1 difference per case, with the M19 5.2% shock-step maximum;
(c) the M14/M15 version comparison for clause-(b) activation counts.

Usage:  python s4_sensitivity.py <stage>
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = sys.argv[1]
D, F, RS = [os.path.join(E, x) for x in ("data", "figures", "results")]
os.makedirs(F, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GI = json.load(open(os.path.join(E, "checks", "gamma_incident.json")))
GP = json.load(open(os.path.join(D, "gamma_rule_probe.json")))
CMP = json.load(open(os.path.join(D, "m17", "s7_full_compare.json")))

INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.2))

# ---- (a) the gamma incident ----------------------------------------------------------------------
ax = axs[0]
fits = {f["rule"]: f for f in GI["fits"]}
keys = ["W1_to_H", "mean", "mass_above_25"]
labels = [r"$W_1(\hat H, H)$", r"$\int x\,d\hat H$", "mass $> 25$"]
true_v = [0.0, GI["regime"]["mean_of_H"], 0.1]
x = np.arange(len(keys)); w = 0.36
ax.bar(x - w / 2, [fits["legacy_N"][k] for k in keys], w, color="#eb6834", edgecolor="k", lw=0.4,
       label=r"legacy $\gamma_0/N$")
ax.bar(x + w / 2, [fits["scale_free"][k] for k in keys], w, color="#2a78d6", edgecolor="k", lw=0.4,
       label=r"scale-free $\gamma_{\rm rel}/\hat R_N$")
for i, (tv, k) in enumerate(zip(true_v, keys)):
    if k != "W1_to_H":
        ax.plot([i - w, i + w], [tv, tv], "k--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.set_yscale("symlog", linthresh=0.05)
ax.set_ylim(-0.05, 60)               # headroom so the legend and the regime note do not sit on the bars
ax.set_ylabel("fitted quantity")
ax.set_title("(a) gamma incident: legacy rule destroys the fit", fontsize=8, loc="left")
ax.legend(fontsize=6, frameon=False, loc="upper left")
ax.text(0.98, 0.86, f"regime: $d$={GI['regime']['d']}, $N$={GI['regime']['N']}, "
        f"$c$={GI['regime']['c']}\n$R(H)$={GI['regime']['R_of_H']:.0f}, "
        f"$\\gamma$ ratio {fits['legacy_N']['gamma']/fits['scale_free']['gamma']:.1e}",
        transform=ax.transAxes, fontsize=5.4, ha="right", va="top", color=INK2)

# ---- (b) the two rules side by side ---------------------------------------------------------------
ax = axs[1]
ks = list(GP.keys())
rel = np.array([GP[k]["rel_diff_W1"] for k in ks]) * 100
short = [k.replace("EXP1_", "").replace("EXP3_", "").replace("_", " ") for k in ks]
cols = ["#eb6834" if "EXP3" in k else "#2a78d6" for k in ks]
ax.barh(np.arange(len(ks)), rel, color=cols, edgecolor="k", lw=0.4)
ax.set_yticks(np.arange(len(ks))); ax.set_yticklabels([k.replace("EXP1_N", "E1 N").replace("EXP3_rep0_", "E3 r0 k")
                                                       .replace("_a", " α=") for k in ks], fontsize=6)
ax.axvline(1.0, color=INK, ls=":", lw=1)
ax.text(1.05, -0.4, "1%", fontsize=6, color=INK)
i_max = int(np.argmax(rel))
ax.annotate(f"shock step k=16: {rel[i_max]:.2f}%", (rel[i_max], i_max), fontsize=6,
            textcoords="offset points", xytext=(-8, -11), ha="right", color=INK)
ax.set_xlabel(r"$|\Delta W_1| / W_1$ between the two $\gamma$ rules  (%)")
ax.set_title("(b) M19: the rules agree except at shocks", fontsize=8, loc="left")
ax.legend(handles=[plt.Line2D([], [], color="#2a78d6", lw=6, label="EXP-1"),
                   plt.Line2D([], [], color="#eb6834", lw=6, label="EXP-3")],
          fontsize=6, frameon=False, loc="lower right")

# ---- (c) M14 / M15 version comparison --------------------------------------------------------------
ax = axs[2]
HS = ["0.03", "0.01", "0.003"]
x = np.arange(len(HS)); w = 0.26
for off, ver, col, lab in [(-w, "preM14", "#9ecae1", "pre-M14"),
                           (0.0, "M14", "#eb6834", "M14"),
                           (w, "M15", "#2a78d6", "M15 (cited)")]:
    tot = [CMP[ver][h]["total"][1] for h in HS]
    ax.bar(x + off, tot, w, color=col, edgecolor="k", lw=0.4, label=lab)
    for xi, v in zip(x + off, tot):
        ax.text(xi, v + 0.7, str(v), ha="center", fontsize=5.8)
ax.set_xticks(x); ax.set_xticklabels([f"$h$={h}" for h in HS], fontsize=7)
ax.set_ylabel("clause (b) activations (of 72)")
ax.set_ylim(0, 82)
ax.set_title("(c) three versions of the same scan", fontsize=8, loc="left")
ax.legend(fontsize=6, frameon=False)
ax.text(0.02, 0.97, "δ̂_fb routine fixed at M17;\nthe cited version is M15", transform=ax.transAxes,
        fontsize=5.6, va="top", color=INK2)

fig.tight_layout()
fig.savefig(os.path.join(F, "figS4_sensitivity.pdf")); fig.savefig(os.path.join(F, "figS4_sensitivity.png"))
plt.close(fig)

summary = dict(
    panels=dict(a="gamma incident (legacy vs scale-free rule)",
                b="two-rule relative W1 difference per case",
                c="M14/M15/pre-M14 activation totals"),
    gamma_incident={f["rule"]: dict(gamma=f["gamma"], W1_to_H=f["W1_to_H"], mean=f["mean"],
                                    mass_above_25=f["mass_above_25"]) for f in GI["fits"]},
    two_rules={k: dict(rel_diff_W1=GP[k]["rel_diff_W1"], gamma_ratio=GP[k]["gamma_ratio"]) for k in GP},
    versions={v: {h: CMP[v][h]["total"] for h in HS} for v in ("preM14", "M14", "M15")},
    MISSING_ITEM=dict(
        item="delta_fb vs N (M24)",
        status="NOT COMPUTED -- no producer and no output file exist in the tree (checked 2026-10-04)",
        ruled="M24: main text says 'no closed-form correction'; S4 carries the measured curve, must include alpha=0.3",
        why_not_drawn=("drawing it would require re-running the delta_fb sweep; interpolating from "
                       "unrelated runs would be a figure that invents the curve it was meant to measure")))
json.dump(summary, open(os.path.join(RS, "s4_sensitivity_summary.json"), "w"), indent=1)

print("=== S4 sensitivity ===")
print("  (a) gamma incident: legacy W1 = %.4f vs scale-free %.4f  (true mean %.1f -> %.3f vs %.3f)"
      % (fits["legacy_N"]["W1_to_H"], fits["scale_free"]["W1_to_H"],
         GI["regime"]["mean_of_H"], fits["legacy_N"]["mean"], fits["scale_free"]["mean"]))
print("  (b) two rules: max rel diff %.4f%% (%s)" % (rel.max(), ks[int(np.argmax(rel))]))
print("  (c) activations h=0.03: preM14 %d / M14 %d / M15 %d"
      % (CMP["preM14"]["0.03"]["total"][1], CMP["M14"]["0.03"]["total"][1], CMP["M15"]["0.03"]["total"][1]))
print("  MISSING: %s -- %s" % (summary["MISSING_ITEM"]["item"], summary["MISSING_ITEM"]["status"]))
