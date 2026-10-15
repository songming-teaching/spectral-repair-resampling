r"""Table I: MH preconditioning efficiency and the target-consistency test -- the REBUILT producer (M20).

The archived tableI.json had no producer in the code (audit); this module is it.  Setup, MH kernel,
functionals, truth and ESS as archive/code_2026-09-27/tableI.py (the producer of the per-regime files
data/tableI/tableI_<regime>.json).  Table I has always used the scale-free gamma (unaffected by M19).

Seeds (M33):
  --seeds legacy   setup key 0 for A_strong else 1  -> reproduces data/tableI/tableI_<regime>.json bitwise
  --seeds m33      config.TABLEI_SETUP_KEY (A_strong 0, G11 1, G12 2, G21 3, B_weak 1): every 2x2 cell its
                   own cloud; A_strong and G11 must come out bitwise-identical to legacy, G12/G21 change.
  MH stream [MASTER, 7, preconditioner index (, 1000 + seedtag)], as before.

Outputs per regime, beyond the historical fields:
  meta.W1_to_own_H   W1(Hhat, H) against the regime's OWN spectrum (0.9 delta_1 + 0.1 delta_kappa) -- D15;
  meta.W1_Hhat       the historical value (against the EXP-1 H0, meaningless) kept only for the bitwise check;
  meta.g, meta.grid_atoms   the inverted weights, so W1 is re-derivable;
  sidak48            the pooled 48-z family (4 preconditioners x 12) Sidak p, ARCHIVED (it used to be printed only).

Also --scan: the W2 r x c coarse scan registered in M33 (I and lift only).

Usage:
  python -m srlab.experiments.tableI --regimes A_strong,G11,G12,G21,B_weak --seeds legacy --out DIR [--workers 20]
  python -m srlab.experiments.tableI --scan --out DIR
"""
import argparse
import json
import os
import sys
import time

from ..threads import pin as _pin
_pin("tableI")                      # D25/M37: 8 threads, archive continuity

import numpy as np                                                          # noqa: E402

from .. import config, laws                                                 # noqa: E402
from ..inverse import Problem, invert, lift_eigs, w_to                      # noqa: E402
from ..stats import ess_geyer, sidak_maxabsz                                # noqa: E402

T = config.TABLEI
NCH, BURN, L = T["NCH"], T["BURN"], T["L"]
PRECS = ("I", "S_r^c + eps I", "lift(H_hat) + eps I", "oracle Sigma + eps I")
FUNCS = ["x[large block]", "x[small block]", "|x|^2/d", "mu-direction",
         "slowest direction (max eig of M^-1/2 C M^-1/2)", "fastest direction (min eig)"]


def regime_params(reg):
    if reg.startswith("scan:"):                                     # scan cell 'scan:<r_obs>:<c>'
        _, r, c = reg.split(":")
        return dict(d=225, c=float(c), kappa=50.0, r_obs=float(r), eps_rel=0.01)
    return T["regimes"][reg]


def setup_key(reg, seeds):
    if reg.startswith("scan:"):
        _, r, c = reg.split(":")
        corner = {(4.0, 0.15): "G11", (200.0, 0.15): "A_strong", (4.0, 0.75): "G12", (200.0, 0.75): "G21"}
        cell = corner.get((float(r), float(c)))
        if cell is not None:          # the four corners ARE the 2x2 cells (M33 W2): same cloud, same numbers
            return config.TABLEI_SETUP_KEY[cell]
        return 100 + int(float(r)) * 10 + int(round(float(c) * 100))   # interior cells: own, distinct keys
    if seeds == "legacy":
        return config.TABLEI_SETUP_KEY_LEGACY[reg]
    return config.TABLEI_SETUP_KEY[reg]


def setup(reg, seeds):
    R = regime_params(reg)
    d = R["d"]; N = int(round(d / R["c"]))
    rng = config.rng("tableI_setup", setup_key(reg, seeds))
    n1 = int(round(0.9 * d))
    s = np.concatenate([np.ones(n1), np.full(d - n1, R["kappa"])])
    EPS = R["eps_rel"] * 1.0
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    mu = np.sqrt(s) * v
    sign = 1 if rng.random() < 0.5 else -1
    r_obs = R["r_obs"]
    xstar = sign * mu + np.sqrt(s) * rng.standard_normal(d)
    y = xstar + np.sqrt(r_obs) * rng.standard_normal(d)
    P = s * r_obs / (s + r_obs)
    means = [P * (mu / s + y / r_obs), P * (-mu / s + y / r_obs)]
    logw = np.array([-0.5 * np.sum((y - m0) ** 2 / (s + r_obs)) for m0 in [mu, -mu]])
    w = np.exp(logw - logw.max())
    w /= w.sum()  # I9: definitional -- two-component posterior weights, exact by construction
    E = w[0] * means[0] + w[1] * means[1]
    signs = rng.choice([-1, 1], N)
    Xc = signs[None, :] * mu[:, None] + np.sqrt(s)[:, None] * rng.standard_normal((d, N))
    lt = rng.gamma(1.0, 1.0, N)
    lam = N * lt / lt.sum()  # I9: definitional self-normalised weights
    n = rng.multinomial(N, lam / N)
    mr = Xc @ n / N
    Sr = (Xc * n[None, :]) @ Xc.T / N - np.outer(mr, mr)
    ev, U = np.linalg.eigh(Sr)
    strong = R["kappa"] > 10
    gr = (np.round(np.exp(np.linspace(np.log(0.05), np.log(4.0 * R["kappa"]), 400)), 8) if strong else config.GRID_005)
    cfg = config.INV_TABLEI_STRONG if strong else config.INV_TABLEI_WEAK
    pr = Problem(np.maximum(ev, 0), d, N, laws.plugin_multi(lam), gr, cfg=cfg)
    g, info = invert(pr)
    lift = lift_eigs(g, gr, d)
    precs = {"I": np.eye(d), "S_r^c + eps I": U @ np.diag(np.maximum(ev, 0) + EPS) @ U.T,
             "lift(H_hat) + eps I": U @ np.diag(lift + EPS) @ U.T, "oracle Sigma + eps I": np.diag(s + EPS)}
    own_H = (np.array([1.0, R["kappa"]]), np.array([0.9, 0.1]))
    meta = dict(zero_atom_emp=float(np.mean(ev < 1e-9 * ev.max())),
                inversion=dict(eps_model=info["eps_model"], nit=info["nit"], success=info["status"] != "failed",
                               status=info["status"], msg=info["msg"]),
                W1_Hhat=w_to(g, gr, np.array([0.5, 2.5]), np.array([0.7, 0.3]), 1),     # historical, wrong reference (D15)
                W1_to_own_H=w_to(g, gr, *own_H, 1), g=g.tolist(), grid_atoms=gr.tolist(),
                post_weights=w.tolist(), grid=(float(gr[0]), float(gr[-1]), len(gr)), gamma=float(pr.gamma),
                setup_key=setup_key(reg, seeds))
    return dict(s=s, mu=mu, y=y, P=P, means=means, w=w, E=E, precs=precs, meta=meta, v=v,
                d=d, N=N, r_obs=r_obs, EPS=EPS, regime=reg, kappa_sigma=R["kappa"])


def logpi(X, S):
    s, mu, y, r_obs = S["s"], S["mu"], S["y"], S["r_obs"]
    q1 = -0.5 * np.sum((X - mu[:, None]) ** 2 / s[:, None], 0)
    q2 = -0.5 * np.sum((X + mu[:, None]) ** 2 / s[:, None], 0)
    return np.logaddexp(q1, q2) - 0.5 * np.sum((y[:, None] - X) ** 2, 0) / r_obs


def functionals(X, S, aw):
    d = S["d"]
    a4 = S["v"] / np.sqrt(S["s"])
    i2, i1 = int(round(0.8 * d)), int(round(0.1 * d))
    return np.stack([X[i2], X[i1], np.sum(X ** 2, 0) / d, a4 @ X, aw[0] @ X, aw[1] @ X])


def post_cov(S):
    P, means, w, E = S["P"], S["means"], S["w"], S["E"]
    return np.diag(P) + sum(wi * np.outer(mi, mi) for wi, mi in zip(w, means)) - np.outer(E, E)


def truth(S, aw):
    d = S["d"]
    P, means, w, E = S["P"], S["means"], S["w"], S["E"]
    a4 = S["v"] / np.sqrt(S["s"])

    def lin(a):
        mean = a @ E
        return mean, a @ (P * a) + sum(wi * (a @ mi) ** 2 for wi, mi in zip(w, means)) - mean ** 2
    e2 = np.zeros(d); e2[int(round(0.8 * d))] = 1
    e1 = np.zeros(d); e1[int(round(0.1 * d))] = 1
    m1, v1 = lin(e2); m2, v2 = lin(e1); m4, v4 = lin(a4)
    m5, v5 = lin(aw[0]); m6, v6 = lin(aw[1])
    m3 = (P.sum() + sum(wi * mi @ mi for wi, mi in zip(w, means))) / d
    ei = [(P.sum() + mi @ mi) / d for mi in means]
    vi = [(2 * np.sum(P ** 2) + 4 * mi @ (P * mi)) / d ** 2 for mi in means]
    v3 = sum(wi * (vv + e ** 2) for wi, vv, e in zip(w, vi, ei)) - m3 ** 2
    return np.array([m1, m2, m3, m4, m5, m6]), np.array([v1, v2, v3, v4, v5, v6])


def run_prec(arg):
    name, reg, seeds = arg[:3]
    lmult = arg[3] if len(arg) > 3 else 1
    seedtag = arg[4] if len(arg) > 4 else 0
    fixedstep = arg[5] if len(arg) > 5 else False
    S = setup(reg, seeds)
    d = S["d"]
    M = S["precs"][name]
    ew, V = np.linalg.eigh(M); Lm = V @ np.diag(np.sqrt(np.maximum(ew, 0))) @ V.T
    Minvh = V @ np.diag(1 / np.sqrt(ew)) @ V.T
    C = post_cov(S)
    lam_pc, Upc = np.linalg.eigh(Minvh @ C @ Minvh)
    aw = []
    for col in (-1, 0):
        a_ = Minvh @ Upc[:, col]; aw.append(a_ / np.sqrt(a_ @ C @ a_))
    kappa_pc = float(lam_pc[-1] / lam_pc[0])
    mtrue, vtrue = truth(S, aw)
    idx = list(S["precs"]).index(name)
    rng = config.rng("tableI_mh", idx) if seedtag == 0 else config.rng("tableI_mh", idx, 1000 + int(seedtag))

    def exact_draws():
        comp = rng.random(NCH) < S["w"][0]
        return (np.where(comp[None, :], S["means"][0][:, None], S["means"][1][:, None])
                + np.sqrt(S["P"])[:, None] * rng.standard_normal((d, NCH)))
    X = exact_draws(); lp = logpi(X, S)
    logstep = np.log(2.38 / np.sqrt(d))
    for it in range(BURN):
        prop = X + np.exp(logstep) * (Lm @ rng.standard_normal((d, NCH)))
        lpp = logpi(prop, S)
        acc = np.log(rng.random(NCH)) < lpp - lp
        X[:, acc] = prop[:, acc]; lp[acc] = lpp[acc]
        logstep += (acc.mean() - T["accept_target"]) * 2.0 / np.sqrt(it + 10)
    if fixedstep:
        X = exact_draws(); lp = logpi(X, S)
    Ltot, thin = L * lmult, lmult
    traces = np.zeros((6, L, NCH), dtype=np.float32)
    s1 = np.zeros((6, NCH)); s2 = np.zeros((6, NCH)); nacc = 0
    t0 = time.time()
    for it in range(Ltot):
        prop = X + np.exp(logstep) * (Lm @ rng.standard_normal((d, NCH)))
        lpp = logpi(prop, S)
        acc = np.log(rng.random(NCH)) < lpp - lp
        X[:, acc] = prop[:, acc]; lp[acc] = lpp[acc]
        nacc += acc.sum()
        f = functionals(X, S, aw)
        s1 += f; s2 += (f - mtrue[:, None]) ** 2
        if it % thin == 0:
            traces[:, it // thin, :] = f
    wall = time.time() - t0
    traces = traces.astype(float)
    ess = np.array([[ess_geyer(traces[f_, :, ch]) for ch in range(NCH)] for f_ in range(6)])
    cm, Tc = s1 / Ltot, s2 / Ltot
    est_m, est_v = cm.mean(1), Tc.mean(1)
    se_m, se_v = cm.std(1, ddof=1) / np.sqrt(NCH), Tc.std(1, ddof=1) / np.sqrt(NCH)
    z = np.concatenate([(est_m - mtrue) / se_m, (est_v - vtrue) / se_v])
    from scipy.stats import skew
    return dict(preconditioner=name, acceptance=float(nacc / (NCH * Ltot)), step=float(np.exp(logstep)),
                wall_sec=wall, L_eff=int(Ltot), lmult=int(lmult), seedtag=int(seedtag), fixedstep=bool(fixedstep),
                ess_thinned_total=ess.sum(1).tolist(), ess_per_step=(ess.sum(1) / (NCH * L)).tolist(),
                min_ess_per_step=float((ess.sum(1) / (NCH * L)).min()),
                est_mean=est_m.tolist(), true_mean=mtrue.tolist(), est_var=est_v.tolist(), true_var=vtrue.tolist(),
                z=z.tolist(), chi2=float(np.sum(z ** 2)), meta=S["meta"], kappa_precond_post=kappa_pc,
                skew_Tc_across_chains=[float(skew(Tc[f_])) for f_ in range(6)], functionals=FUNCS)


def _keyed(pair):
    """(label, args) -> (label, row), executed IN THE WORKER.

    See exp1._keyed: building `(label, run_prec(t))` in the parent would run the work in the PARENT
    and serialise the Pool.  Module-level so it is picklable."""
    label, args = pair
    return label, run_prec(args)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="A_strong,G11,G12,G21,B_weak")
    ap.add_argument("--seeds", choices=("legacy", "m33"), default="m33")
    ap.add_argument("--scan", action="store_true", help="the M33 W2 r x c scan (I and lift only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    if a.scan:
        regs = [f"scan:{r:g}:{c:g}" for r in (4, 20, 60, 200) for c in (0.15, 0.35, 0.55, 0.75)]
        tasks = [(p, reg, a.seeds) for reg in regs for p in ("I", "lift(H_hat) + eps I")]
    else:
        regs = a.regimes.split(",")
        tasks = [(p, reg, a.seeds) for reg in regs for p in PRECS]
    os.makedirs(a.out, exist_ok=True)
    from multiprocessing import Pool
    from ..runbudget import drive, budget_status
    budget = config.run_budget("tableI")
    t0 = time.time(); res = {}
    with Pool(config.workers(a.workers)) as pool:
        # stage F: the run-level budget wraps the loop.  The stream is `pool.imap_unordered` over
        # (_label, args) pairs and `_keyed` runs IN THE WORKER, so the Pool's parallelism is exactly
        # what it was -- see drive()'s docstring for why building (label, run_prec(t)) here is wrong.
        pairs = pool.imap_unordered(_keyed, [(f"{t[1]}|{t[0]}", t) for t in tasks])
        for tk, row in drive(budget, pairs, a.out):
            reg, prec = tk.split("|", 1)
            res.setdefault(reg, []).append(row)
    for reg, rows in res.items():
        rows.sort(key=lambda r: [p for p in PRECS].index(r["preconditioner"]))
        pooled = [z for r in rows for z in r["z"]]
        summary = dict(regime=reg, seeds=a.seeds, n_z=len(pooled), sidak_pooled=sidak_maxabsz(pooled),
                       sidak_family="pooled over all preconditioners in this file x 12 functionals (M20)")
        fname = f"tableI_{reg.replace(':', '_')}.json"
        with open(os.path.join(a.out, fname), "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        with open(os.path.join(a.out, fname.replace(".json", "_sidak.json")), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
    status = budget_status(budget)
    print(f"Table I ({'scan' if a.scan else a.regimes}, seeds={a.seeds}) {len(tasks)} runs, "
          f"{time.time() - t0:.0f}s -> {a.out} [{status}]")
    return 2 if status == "budget_exceeded" else 0


if __name__ == "__main__":
    sys.exit(main())
