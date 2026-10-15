r"""s4b_dfb_vs_N.py -- S4 item (3): delta_fb versus N at fixed c = 0.75 (M24 execution, 2026-10-05).

Reads the rows of srlab.experiments.dfb_n and draws the three panels of EXPERIMENT_PLAN_SUPPLEMENT §1.4:
  (a) delta_fb vs N at the deployed bandwidth h_rel = 0.03, alpha = 10 and alpha = 0.3, median and IQR,
      with the log-log slope and its 95% CI (bootstrap over reps within each N: 2000 draws, seed
      20260925 -- the Fig.1(c) convention of checks/figure_additions.py);
  (b) the I6 resolution ratio dx/eta of the delta_fb quantile call vs N (worst over reps), with the
      cap line at 1.0 -- the direct answer to "is the cap hit, and does that move the curve?";
  (c) bandwidth sensitivity: delta_fb median vs N for all three h_rel, both alphas.
Abstained (cell, h) entries are COUNTED and printed on the figure, never dropped silently.
Also writes <rows_dir>/dfb_N_summary.json (medians, IQRs, slopes + CIs, cap statistics, verdict).

Usage:  python scripts/s4b_dfb_vs_N.py <window>/dfb_N [out_fig_dir=figures]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = sys.argv[1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures")
os.makedirs(F, exist_ok=True)
rows = json.load(open(os.path.join(RD, "dfb_N_rows.json"), encoding="utf-8"))

HRELS = ("0.03", "0.01", "0.003")
DEP = "0.03"
cells = sorted({(r["N"], r["alpha"]) for r in rows})
alphas = sorted({a for _, a in cells}, reverse=True)


def vals(N, al, h, key="delta_fb"):
    out = []
    for r in rows:
        if r["N"] == N and r["alpha"] == al:
            e = r["by_h"].get(h)
            if e and not e.get("abstain"):
                out.append(e[key])
    return np.array(out, float)


def cap_ratio(e):
    """forward.law_quantiles info['cap']['worst_dx_over_eta'] (M26 mandatory field)."""
    v = (e.get("cap") or {}).get("worst_dx_over_eta")
    return np.nan if v is None else float(v)


def cap_exceed(e):
    """forward.law_quantiles info['cap']['exceeded'] (bool)."""
    return int(bool((e.get("cap") or {}).get("exceeded", False)))


rng = np.random.default_rng(20260925)


def slope_ci(Ns, al, h):
    ys = [vals(N, al, h) for N in Ns]
    if any(len(y) == 0 for y in ys) or len(Ns) < 2:
        return None
    s0 = float(np.polyfit(np.log(Ns), np.log([np.median(y) for y in ys]), 1)[0])
    bs = []
    for _ in range(2000):
        bs.append(float(np.polyfit(np.log(Ns), np.log([np.median(rng.choice(y, len(y))) for y in ys]), 1)[0]))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return dict(slope=s0, ci95=[float(lo), float(hi)], Ns=list(map(int, Ns)))


summary = dict(source=os.path.join(RD, "dfb_N_rows.json"), deployed_h=DEP, cells={}, slopes={}, abstained={})
for (N, al) in cells:
    for h in HRELS:
        v = vals(N, al, h)
        ents = [r["by_h"].get(h) for r in rows if r["N"] == N and r["alpha"] == al]
        n_abs = sum(1 for e in ents if e and e.get("abstain"))
        live = [e for e in ents if e and not e.get("abstain")]
        summary["cells"][f"N={N}|a={al}|h={h}"] = dict(
            n=int(len(v)), abstained=n_abs,
            median=float(np.median(v)) if len(v) else None,
            q25=float(np.quantile(v, .25)) if len(v) else None, q75=float(np.quantile(v, .75)) if len(v) else None,
            worst_dx_over_eta=float(np.nanmax([cap_ratio(e) for e in live])) if live else None,
            cap_exceed_count=int(sum(cap_exceed(e) for e in live)))
        if n_abs:
            summary["abstained"][f"N={N}|a={al}|h={h}"] = n_abs
for al in alphas:
    Ns = sorted(N for N, a in cells if a == al)
    for h in HRELS:
        s = slope_ci(Ns, al, h)
        if s:
            summary["slopes"][f"a={al}|h={h}"] = s

# M24 execution-record verdict (rules registered before computation)
sh, sl = summary["slopes"].get(f"a=0.3|h={DEP}"), summary["slopes"].get(f"a=10.0|h={DEP}")
if sh and sl:
    meds = [summary["cells"][f"N={N}|a=0.3|h={DEP}"]["median"] for N in sh["Ns"]]
    mono = all(b < a for a, b in zip(meds, meds[1:]))
    zero_in_ci = sh["ci95"][0] <= 0.0 <= sh["ci95"][1]
    if zero_in_ci:
        verdict = "slope ~ 0 (95% CI contains 0): bias does NOT decay with N -> intrinsic discretisation effect"
    elif mono and sh["slope"] <= sl["slope"]:
        verdict = "main criterion MET: monotone decrease and heavy-tail slope at least as steep -> finite-N effect"
    else:
        verdict = (f"main criterion NOT met: monotone={mono}, slope(0.3)={sh['slope']:+.3f} vs "
                   f"slope(10)={sl['slope']:+.3f}; report as measured")
    exc_all = all(summary["cells"][f"N={N}|a=0.3|h={DEP}"]["cap_exceed_count"] > 0 for N in sh["Ns"])
    summary["verdict"] = dict(text=verdict, monotone_heavy=mono, cap_exceeded_at_every_N_heavy=exc_all)
json.dump(summary, open(os.path.join(RD, "dfb_N_summary.json"), "w", encoding="utf-8"), indent=1)

# ---------------------------------------------------------------------------------------------- figure
INK, INK2 = "#0b0b0b", "#52514e"
COL = {10.0: "#2a78d6", 3.0: "#7aa64a", 1.0: "#a0712b", 0.3: "#eb6834"}
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
allN = sorted({N for N, _ in cells})

ax = axs[0]
for al in (10.0, 0.3):
    Ns = sorted(N for N, a in cells if a == al)
    st = [summary["cells"][f"N={N}|a={al}|h={DEP}"] for N in Ns]
    ok = [i for i, s in enumerate(st) if s["median"] is not None]
    x = [Ns[i] for i in ok]
    ax.plot(x, [st[i]["median"] for i in ok], "-o", ms=3, color=COL[al])
    ax.fill_between(x, [st[i]["q25"] for i in ok], [st[i]["q75"] for i in ok], color=COL[al], alpha=0.18, lw=0)
    s = summary["slopes"].get(f"a={al}|h={DEP}")
    lab = f"α={al:g}" + (f": slope {s['slope']:+.2f} [{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}]" if s else "")
    ax.plot([], [], "-o", ms=3, color=COL[al], label=lab)
ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off(); ax.set_xticks(allN); ax.set_xticklabels(allN)
ax.set_xlabel("N  (c = 0.75 fixed, d = cN)"); ax.set_ylabel(r"$\hat\delta_{\rm fb}$  (median, IQR band)")
ax.set_title(f"(a) δ̂_fb vs N, deployed h_rel = {DEP}", fontsize=8, loc="left")
ax.legend(fontsize=6, frameon=False)

ax = axs[1]
for al in alphas:
    Ns = sorted(N for N, a in cells if a == al)
    y = [summary["cells"][f"N={N}|a={al}|h={DEP}"]["worst_dx_over_eta"] for N in Ns]
    ax.plot(Ns, [np.nan if v is None else v for v in y], "-o", ms=3, color=COL[al], label=f"α={al:g}")
ax.axhline(1.0, color="k", ls="--", lw=0.9)
ax.text(allN[0], 1.0, " cap: dx = η", fontsize=6, va="bottom", color=INK2)
ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(allN); ax.set_xticklabels(allN)
ax.set_xlabel("N"); ax.set_ylabel("worst dx/η over reps")
ax.set_title("(b) I6 resolution of the δ̂_fb call (>1 = cap hit)", fontsize=8, loc="left")
ax.legend(fontsize=6, frameon=False)

ax = axs[2]
LS = {"0.03": "-", "0.01": "--", "0.003": ":"}
for al in (10.0, 0.3):
    Ns = sorted(N for N, a in cells if a == al)
    for h in HRELS:
        y = [summary["cells"][f"N={N}|a={al}|h={h}"]["median"] for N in Ns]
        ax.plot(Ns, [np.nan if v is None else v for v in y], LS[h], marker="o", ms=2.5, color=COL[al],
                label=f"α={al:g}, h={h}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off(); ax.set_xticks(allN); ax.set_xticklabels(allN)
ax.set_xlabel("N"); ax.set_ylabel(r"median $\hat\delta_{\rm fb}$")
ax.set_title("(c) bandwidth sensitivity", fontsize=8, loc="left")
ax.legend(fontsize=5.5, frameon=False, ncol=2)

n_abs = sum(summary["abstained"].values())
fig.text(0.01, 0.005, f"R = 20 reps per cell; multinomial; plug-in law; H = H₀; abstained (cell×h): {n_abs}"
         + ("" if not n_abs else "  — " + ", ".join(f"{k}: {v}" for k, v in summary["abstained"].items())),
         fontsize=5.6, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(F, "figS4b_dfb_vs_N.pdf")); fig.savefig(os.path.join(F, "figS4b_dfb_vs_N.png"))
plt.close(fig)
print(json.dumps(dict(slopes=summary["slopes"], verdict=summary.get("verdict"), abstained=summary["abstained"]),
                 indent=1, ensure_ascii=False))
