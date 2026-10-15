"""S3 analysis: pre-registered verdicts P1-P7 (M16) + supplementary figure figS3_t1a.

Usage: python s3_analyze.py <expdir>
"""
import os, sys, json
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srcore import stieltjes, law_zero_atom
from s3_t1a import LAW, mixpoi_pmf, N

E = sys.argv[1]
D = os.path.join(E, "data", "s3")
T = json.load(open(os.path.join(D, "s3_prereg_targets.json")))
ROWS = json.load(open(os.path.join(D, "s3_rows.json")))
TH = {r["name"]: r for r in T["laws"]}
BY = {}
for r in ROWS:
    BY.setdefault(r["law"], []).append(r)
CS = T["CS"]; C0 = 0.75
A1, M2H = T["H0"]["a1"], T["H0"]["m2"]

def col(law, key): return np.array([r[key] for r in BY[law]], float)
def ms(law, key):
    x = col(law, key); return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))

CHAINS = {"gamma": [f"gamma_{a:g}" for a in [100, 30, 10, 3, 1, 0.5, 0.3]],
          "lognorm": [f"lognorm_{v:g}" for v in [0.25, 1, 4]],
          "sym": [f"sym_{s:g}" for s in [0, 0.25, 0.5, 0.75, 1]]}
GRP = T["same_r2_group"]
V, OUT = {}, {}

# ---- P1 convex-order monotonicity of p_s --------------------------------------------------------
p1 = {}
for ch, laws in CHAINS.items():
    th = [TH[l]["p_s"] for l in laws]; em = [ms(l, "ps_multi") for l in laws]
    dec_th = all(np.diff(th) < 0); dec_em = all(np.diff([m for m, _ in em]) < 0)
    tol = [abs(m - t) <= 3 * s + 0.002 for (m, s), t in zip(em, th)]
    p1[ch] = dict(laws=laws, p_s=th, emp=[m for m, _ in em], se=[s for _, s in em],
                  strictly_decreasing_theory=dec_th, strictly_decreasing_emp=dec_em, within_tol=tol)
V["P1"] = all(v["strictly_decreasing_theory"] and v["strictly_decreasing_emp"] and all(v["within_tol"])
              for v in p1.values())
OUT["P1"] = p1

# ---- P2 upper bound 1 - 1/e ---------------------------------------------------------------------
up = T["ps_upper"]
p2 = {l: dict(p_s=TH[l]["p_s"], emp=ms(l, "ps_multi")) for l in TH}
V["P2"] = all(v["p_s"] <= up + 1e-12 and v["emp"][0] <= up + 3 * v["emp"][1] + 0.002 for v in p2.values()) \
    and abs(TH["sym_0"]["p_s"] - up) < 1e-12 and all(TH[l]["p_s"] < up - 1e-4 for l in TH if l != "sym_0")
OUT["P2"] = dict(upper=up, sym0=TH["sym_0"]["p_s"], max_other=max(TH[l]["p_s"] for l in TH if l != "sym_0"))

# ---- P3 zero-atom law (delta_1) -----------------------------------------------------------------
p3, ok3 = {}, True
for l in TH:
    for c in CS:
        d = int(round(c * N))
        ps = col(l, "ps_multi"); at = col(l, f"atom_d1_multi_c{c}")
        ident = np.maximum(0, 1 - (N * ps - 1) / d)
        rank_ok = float(np.max(np.abs(at - ident)))
        m, s = ms(l, f"atom_d1_multi_c{c}"); th = TH[l][f"atom_multi_c{c}"]
        good = rank_ok < 1.5 / d and abs(m - th) <= 3 * s + 0.003
        ok3 &= good
        p3[f"{l}|c{c}"] = dict(emp=m, se=s, theory=th, rank_identity_maxdev=rank_ok, ok=good)
V["P3"] = bool(ok3); OUT["P3"] = p3

# ---- P4 same-r2 counterexample -----------------------------------------------------------------
p4 = {}
m2 = {l: ms(l, f"m2_d1_multi_c{C0}") for l in GRP}
# conditional target for the delta_1 branch: 1 + b2 (c + 1/N) (uncentred, a1 = m2H = 1)
cond_dev = {}
for l in GRP:
    y = col(l, f"m2u_d1_multi_c{C0}") - (1 + col(l, "b2_multi") * (C0 + 1.0 / N))   # uncentred (M16 addendum)
    cond_dev[l] = (float(y.mean()), float(y.std(ddof=1) / np.sqrt(len(y))))
m2_means = np.array([m2[l][0] for l in GRP])
atoms = {l: col(l, f"atom_d1_multi_c{C0}") for l in GRP}
order = sorted(GRP, key=lambda l: TH[l]["p_s"])
adj_p = {f"{a}|{b}": float(mannwhitneyu(atoms[a], atoms[b]).pvalue) for a, b in zip(order[:-1], order[1:])}
at_means = np.array([atoms[l].mean() for l in GRP])
c05 = {l: ms(l, "atom_d1_multi_c0.5") for l in GRP}
limW = {k: v for k, v in T["limit_W1_same_r2"].items() if "m2_numeric" not in k}
p4 = dict(m2=m2, m2_rel_range=float(np.ptp(m2_means) / m2_means.mean()),
          cond_dev=cond_dev, atom_means=dict(zip(GRP, at_means.tolist())), atom_range=float(np.ptp(at_means)),
          adjacent_MWU=adj_p, atom_c05=c05, limit_W1=limW,
          limit_m2_numeric={k: v for k, v in T["limit_W1_same_r2"].items() if "m2_numeric" in k})
V["P4"] = (all(abs(m) <= 3 * s for m, s in cond_dev.values()) and p4["m2_rel_range"] < 0.01
           and p4["atom_range"] >= 0.15 and all(p < 1e-6 for p in adj_p.values())
           and c05["sym_1"][0] > 0 and c05["spike_0.1"][0] == 0 and c05["spike_0.02"][0] == 0
           and all(v > 0 for v in limW.values()))
OUT["P4"] = p4

# ---- P5 order reversal --------------------------------------------------------------------------
a, b = "spike_0.02", "sym_0.75"
p5 = dict(r2=(TH[a]["r2"], TH[b]["r2"]), p_s=(TH[a]["p_s"], TH[b]["p_s"]),
          emp=(ms(a, "ps_multi"), ms(b, "ps_multi")),
          MWU_p=float(mannwhitneyu(col(a, "ps_multi"), col(b, "ps_multi"), alternative="greater").pvalue),
          atom=(ms(a, f"atom_d1_multi_c{C0}"), ms(b, f"atom_d1_multi_c{C0}")))
V["P5"] = p5["r2"][0] > p5["r2"][1] and p5["emp"][0][0] > p5["emp"][1][0] and p5["MWU_p"] < 1e-6 \
    and p5["atom"][0][0] < p5["atom"][1][0]
OUT["P5"] = p5

# ---- P6 residual <= multinomial (H0 branch) -----------------------------------------------------
p6, ok6 = {}, True
for l in TH:
    mm, rr = col(l, "m2_H0_multi"), col(l, "m2_H0_res")
    dif = col(l, "m2u_H0_multi") - col(l, "m2u_H0_res")   # conditional check on uncentred (M16 addendum)
    dth = col(l, "m2_H0_multi_condth") - col(l, "m2_H0_res_condth")
    e = dif - dth
    wtd = col(l, "m2_H0_weighted")
    ex_m, ex_r = mm - wtd, rr - wtd
    unit = C0 * A1 ** 2
    good = TH[l]["E_pi_U"] <= 1 and rr.mean() <= mm.mean() and abs(e.mean()) <= 3 * e.std(ddof=1) / np.sqrt(len(e))
    ok6 &= good
    p6[l] = dict(E_pi_U=TH[l]["E_pi_U"], m2_multi=float(mm.mean()), m2_res=float(rr.mean()), diff_centred=float((mm - rr).mean()),
                 diff=float(dif.mean()), diff_condth=float(dth.mean()), diff_se=float(dif.std(ddof=1) / np.sqrt(len(dif))),
                 frac_res_below_multi=float(np.mean(rr < mm)),
                 excess_multi=(float(ex_m.mean() / unit), float(ex_m.std(ddof=1) / np.sqrt(len(ex_m)) / unit)),
                 excess_res=(float(ex_r.mean() / unit), float(ex_r.std(ddof=1) / np.sqrt(len(ex_r)) / unit)), ok=good)
sym1_eq = float(np.max(np.abs(col("sym_1", "m2_H0_res") - col("sym_1", "m2_H0_weighted"))))
p6["_sym1_res_minus_weighted_maxabs"] = sym1_eq
V["P6"] = bool(ok6) and sym1_eq < 1e-10
OUT["P6"] = p6

# ---- P7 linearity in r2 -------------------------------------------------------------------------
d = int(round(C0 * N)); d0 = int(round(0.7 * d))
sig = np.where(np.arange(d) < d0, 0.5, 2.5); cN = d / N
slope_th = cN * sig.mean() ** 2 + (sig ** 2).mean() / N
r2 = np.array([r["r2N"] for r in ROWS]); EU = np.array([r["EU_N"] for r in ROWS])
p7 = {"slope_theory": slope_th}
for sc, y in [("weighted", np.array([r["m2_H0_weighted"] for r in ROWS])),
              ("multi", np.array([r["m2_H0_multi"] for r in ROWS])),
              ("res", np.array([r["m2_H0_res"] for r in ROWS]) - slope_th * EU)]:
    s, i = np.polyfit(r2, y, 1)
    p7[sc] = dict(slope=float(s), intercept=float(i), rel_err=float(abs(s / slope_th - 1)),
                  resid_sd=float(np.std(y - (s * r2 + i))))
p7["intercept_theory"] = dict(weighted=M2H, multi=M2H + C0 * A1 ** 2, res=M2H)
V["P7"] = all(p7[s]["rel_err"] < 0.02 for s in ["weighted", "multi", "res"])
OUT["P7"] = p7
OUT["verdicts"] = V
OUT["lower_envelope_check"] = T["lower_envelope_check"]
json.dump(OUT, open(os.path.join(E, "results", "s3_summary.json"), "w"), indent=1, default=float)
print(json.dumps(V, indent=1))

# ============================== figure =========================================================
COL = {"gamma": "#2a78d6", "lognorm": "#eb6834", "sym": "#1baf7a", "spike": "#eda100"}
MK = {"gamma": "o", "lognorm": "s", "sym": "^", "spike": "D"}
LBL = {"gamma": "Gamma($\\alpha$)", "lognorm": "log-normal", "sym": "sym. two-point", "spike": "spike ($r_2{=}2$)"}
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))

# (a) p_s vs r2 with envelope
ax = axs[0]
rr = np.linspace(1, 5.2, 300)
ax.fill_between(rr, 0.15, (1 - np.exp(-rr)) / rr, color="#f0efec", lw=0)
ax.fill_between(rr, up, 0.7, color="#f0efec", lw=0)
ax.plot(rr, (1 - np.exp(-rr)) / rr, color=INK2, lw=1, ls="--")
ax.axhline(up, color=INK2, lw=1, ls="--")
ax.text(4.9, up + 0.006, "$1-e^{-1}$", ha="right", va="bottom", color=INK2)
ax.text(3.25, 0.262, "lower envelope\n$(1-e^{-r_2})/r_2$", ha="left", va="top", color=INK2, fontsize=7)
ax.axhline(0.5, color=INK2, lw=0.6, ls=":")
ax.text(5.15, 0.503, "$c=0.5$: zero atom iff $p_s<c$", color=INK2, va="bottom", ha="right", fontsize=6.5)
ax.text(1.03, 0.205, "($c=0.75>1-e^{-1}$: every law has a zero atom)", fontsize=6, color=INK2)
# closed-form chain curves
al = np.logspace(-0.6, 2.2, 200); ax.plot(1 + 1 / al, 1 - (al / (al + 1)) ** al, color=COL["gamma"], lw=1.5)
xg, wg = np.polynomial.hermite_e.hermegauss(120); wg = wg / wg.sum()
vv = np.linspace(0.02, 4.2, 60); sg = np.sqrt(np.log1p(vv))
ax.plot(1 + vv, [1 - wg @ np.exp(-np.exp(-0.5 * q * q + q * xg)) for q in sg], color=COL["lognorm"], lw=1.5)
s_ = np.linspace(0, 1, 100); ax.plot(1 + s_ ** 2, 1 - np.exp(-1) * np.cosh(s_), color=COL["sym"], lw=1.5)
for fam in ["gamma", "lognorm", "sym", "spike"]:
    laws = [l for l in TH if TH[l]["fam"] == fam]
    x = [TH[l]["r2"] for l in laws]; y = [ms(l, "ps_multi")[0] for l in laws]
    ax.plot(x, y, MK[fam], ms=5, mfc=COL[fam], mec="#fcfcfb", mew=1, label=LBL[fam], ls="none")
ax.annotate("same $r_2$,\ndifferent $p_s$", xy=(1.97, 0.45), xytext=(1.2, 0.33), fontsize=7, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xlim(1, 5.2); ax.set_ylim(0.2, 0.67)
ax.set_xlabel("$r_2=\\mathbb{E}\\Lambda^2$"); ax.set_ylabel("active-column fraction $p_s$")
ax.set_title("(a) $p_s$ is ordered by $\\preceq_{\\rm cx}$, not by $r_2$", fontsize=8, loc="left")
ax.legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.95))

# (b) same-r2 ESDs, delta_1, c = 0.75
ax = axs[1]
Z = np.load(os.path.join(D, "s3_esd.npz"))
xs = np.linspace(1e-3, 9, 1500)
for l, fam in [("sym_1", "sym"), ("spike_0.02", "spike")]:
    ev = np.concatenate([Z[k] for k in Z.files if k.startswith(l + "_") and "_multi_" in k])
    pos = ev[ev > 1e-9 * ev.max()]
    nrep = sum(1 for k in Z.files if k.startswith(l + "_") and "_multi_" in k)
    w = np.full(len(pos), 1.0 / (nrep * int(round(C0 * N))))
    ax.hist(pos, bins=np.linspace(0, 9, 91), weights=w / 0.1, histtype="step", color=COL[fam], lw=1.2,
            label=("$\\{0,2\\}$" if fam == "sym" else "$0.98\\delta_{6/7}{+}0.02\\delta_8$") + " (empirical)")
    nu = mixpoi_pmf(LAW[l]); t = np.arange(len(nu), dtype=float); one = np.array([1.0])
    p0 = law_zero_atom(one, one, nu[0], C0)
    m, *_ = stieltjes(xs + 2e-3j, one, one, t, nu, C0)
    ax.plot(xs, np.maximum((m + p0 / (xs + 2e-3j)).imag, 0) / np.pi, color=COL[fam], lw=0.8, ls="--")
    am = OUT["P4"]["atom_means"][l]; m2e = OUT["P4"]["m2"][l][0]
    ax.text(0.97, 0.93 if fam == "sym" else 0.78,
            f"{'{0,2}' if fam == 'sym' else '$0.98\\delta_{6/7}{+}0.02\\delta_8$'}\n"
            f"atom {am:.3f} (th. {TH[l]['atom_multi_c0.75']:.3f}),  $\\hat m_2$ {m2e:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK)
ax.plot([], [], color=INK2, lw=0.8, ls="--", label="limit law (theory)")
ax.legend(frameon=False, fontsize=6.5, loc="center right")
ax.set_xlim(0, 9); ax.set_ylim(0, None)
ax.set_xlabel("eigenvalue (nonzero part)"); ax.set_ylabel("density")
ax.set_title(f"(b) $r_2=2$, $H=\\delta_1$, $c={C0}$: equal $m_2$, different law", fontsize=8, loc="left")

# (c) scheme excess over the weighted ensemble, in units of c a1^2
ax = axs[2]
ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--"); ax.axhline(1, color=INK2, lw=0.8, ls="--")
ax.text(0.02, 1.02, "multinomial: $ca_1^2$", color=INK2, fontsize=7, va="bottom")
ax.text(0.47, 0.60, "residual: $ca_1^2\\,\\mathbb{E}_\\pi U$", color=INK2, fontsize=7, rotation=33)
for l in TH:
    fam = TH[l]["fam"]; x = TH[l]["E_pi_U"]
    ax.errorbar(x, OUT["P6"][l]["excess_multi"][0], OUT["P6"][l]["excess_multi"][1], fmt=MK[fam], ms=4,
                mfc="none", mec=COL[fam], ecolor=COL[fam], elinewidth=0.8)
    ax.errorbar(x, OUT["P6"][l]["excess_res"][0], OUT["P6"][l]["excess_res"][1], fmt=MK[fam], ms=4,
                mfc=COL[fam], mec="#fcfcfb", ecolor=COL[fam], elinewidth=0.8)
ax.set_xlim(-0.03, 0.9); ax.set_ylim(-0.08, 1.15)
ax.set_xlabel("$\\mathbb{E}_\\pi U$ (closed form)")
ax.set_ylabel("$(\\hat m_2^{\\rm scheme}-\\hat m_2^{\\rm weighted})/(ca_1^2)$")
ax.set_title("(c) residual $\\leq$ multinomial ($H_0$, $c=0.75$)", fontsize=8, loc="left")
ax.text(0.98, 0.3, "hollow: multinomial\nfilled: residual", transform=ax.transAxes, ha="right", fontsize=6.5, color=INK2)
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(E, "figures", f"figS3_t1a.{ext}"), dpi=200)
print("figure written")
