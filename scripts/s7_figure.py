"""S7 figure + summary: activation region of supervisor clause (b).

Two panels:
 (a) activation map over (c, alpha) at the DEPLOYED bandwidth h = 0.03,
     colour/marker = median ratio Bhat/delta_fb;
 (b) bandwidth sensitivity: activation fraction vs c for h in {0.03, 0.01, 0.003},
     showing that the boundary is numerical as well as statistical.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = sys.argv[1]
D, F, RS = [os.path.join(E, x) for x in ("data", "figures", "results")]
# argv: <expdir> [source json in data/s7] [output tag]; default = the historical v2 figure.
SRC = sys.argv[2] if len(sys.argv) > 2 else "s7_activation_v2.json"
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
R = json.load(open(os.path.join(D, "s7", SRC)))
if "p0_distortion" not in R[0]:          # M15 rows do not repeat the cell descriptors: take them from M14
    M14 = {(r["c"], r["alpha"], r["rep"]): r for r in json.load(open(os.path.join(D, "s7", "s7_activation_v3_M14.json")))}
    for r in R:
        r["p0_distortion"] = M14[(r["c"], r["alpha"], r["rep"])]["p0_distortion"]
CS = sorted(set(r["c"] for r in R)); ALS = sorted(set(r["alpha"] for r in R))
HRELS = ["0.03", "0.01", "0.003"]
DEPLOYED = "0.03"

def med(c, a, h, key):
    return float(np.median([r["by_h"][h][key] for r in R if r["c"] == c and r["alpha"] == a]))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axs[0]
X, Y, C, S = [], [], [], []
for c in CS:
    for a in ALS:
        X.append(c); Y.append(np.log10(a)); C.append(med(c, a, DEPLOYED, "ratio"))
        S.append(90 if med(c, a, DEPLOYED, "ratio") > 1 else 22)
sc = ax.scatter(X, Y, c=C, s=S, cmap="RdYlBu_r", edgecolor="k", lw=0.5,
                norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=2.8))
# smooth depiction of the 1.0 level (the raw grid is noisy with 3 reps/cell)
from scipy.interpolate import RectBivariateSpline
Zg = np.array([[np.log10(med(c, a, DEPLOYED, "ratio")) for c in CS] for a in ALS])
spl = RectBivariateSpline(np.log10(np.array(ALS)), np.array(CS), Zg, kx=2, ky=2, s=0.6)
cx = np.linspace(min(CS), max(CS), 200); cy = np.linspace(min(np.log10(ALS)), max(np.log10(ALS)), 200)
cs = ax.contour(cx, cy, spl(cy, cx), levels=[0.0], colors="k", linewidths=1.6, linestyles="--")
ax.clabel(cs, fmt={0.0: "clause (b) boundary"}, fontsize=7)
plt.colorbar(sc, ax=ax, label="median  B_hat / delta_fb", extend="both")
ax.set_xticks(CS); ax.set_xticklabels([f"{c:g}" for c in CS], fontsize=7)
ax.set_yticks(np.log10(ALS)); ax.set_yticklabels([f"{a:g}" for a in ALS], fontsize=7)
ax.set_xlabel("c = d/N"); ax.set_ylabel("weight-law alpha")
ax.set_title(f"(a) clause (b) activation, deployed bandwidth h={DEPLOYED}\n"
             "large markers: activated (>1)" + (f"   [{TAG}]" if TAG else ""), fontsize=7.5)

ax = axs[1]
for h, mk, col in zip(HRELS, ["o", "s", "^"], ["C0", "C1", "C2"]):
    frac = [np.mean([r["by_h"][h]["activated"] for r in R if r["c"] == c]) for c in CS]
    ax.plot(CS, frac, mk + "-", color=col, ms=4, lw=1,
            label=f"h={h}" + ("  (deployed)" if h == DEPLOYED else ""))
ax.axhline(0.5, color="grey", ls=":", lw=0.8)
ax.set_xlabel("c = d/N"); ax.set_ylabel("activated fraction")
ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)
ax.set_title("(b) the boundary is bandwidth-dependent\n$\\hat\\varepsilon$ is an absolute"
             " discrepancy; smaller h inflates it", fontsize=7.5)
OUTN = "figS_s7_activation" + (f"_{TAG}" if TAG else "")
fig.tight_layout(); fig.savefig(os.path.join(F, OUTN + ".pdf"))
fig.savefig(os.path.join(F, OUTN + ".png")); plt.close(fig)

summary = {"deployed_h": DEPLOYED,
           "activation_fraction_by_c": {str(c): float(np.mean([r["by_h"][DEPLOYED]["activated"]
                                                               for r in R if r["c"] == c])) for c in CS},
           "ratio_median_by_c": {str(c): med(c, ALS[0], DEPLOYED, "ratio") * 0 + float(np.median(
               [r["by_h"][DEPLOYED]["ratio"] for r in R if r["c"] == c])) for c in CS},
           "kappa_median_by_c": {str(c): float(np.median([r["by_h"][DEPLOYED]["kappa"]
                                                          for r in R if r["c"] == c])) for c in CS},
           "zero_atom_by_c": {str(c): float(np.median([r["p0_distortion"] for r in R if r["c"] == c]))
                              for c in CS},
           "activation_fraction_by_h": {h: float(np.mean([r["by_h"][h]["activated"] for r in R]))
                                        for h in HRELS},
           "exp3_setting": {"c": 0.75, "alpha_regime": 3.0,
                            "ratio_h003": med(0.75, 3.0, DEPLOYED, "ratio"),
                            "activated": bool(med(0.75, 3.0, DEPLOYED, "ratio") > 1)}}
json.dump(summary, open(os.path.join(RS, "s7_summary" + (f"_{TAG}" if TAG else "") + ".json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
