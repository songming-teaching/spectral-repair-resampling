r"""EXP-2: mismatch certificate and identifiability -- migrated onto the srlab core (P3).

Same design, draw order and seed streams as archive/code_2026-09-27/exp2.py (producer of
experiment/data/exp2/exp2_rows.json):

  part a  N = 2000, d = 1500 (d0 = 1050), R = 50, groups base / endo10 / endo750 / corr0.5 / corr0.9
          -> settings S1-S5 (+ S4c permuted controls); stream exp2a = [MASTER, 2, rep, group index]
  part b  H = delta_1, N = 1000, c in {0.5, 0.75, 1, 1.25} x alpha in {10, 3, 1, 0.5, 0.3}, R = 20;
          stream exp2b = [MASTER, 3, int(100 c), int(10 alpha), rep]

  --rule baseline  frozen pre-M19 gamma (must reproduce the archive bitwise)
  --rule m19       scale-free gamma (window rerun)

Usage: python -m srlab.experiments.exp2 --rule baseline --out DIR [--subset a:0-1 | b:0-1] [--workers 1]
"""
import argparse
import json
import os
import sys
import time

from ..threads import pin as _pin
_pin("default")                     # D25/M37: forced before numpy loads

import numpy as np                                                     # noqa: E402

from .. import config, laws, sampling                                  # noqa: E402
from ..inverse import Problem, invert, kappa_hat, w_to                 # noqa: E402

GRID = config.GRID_005
GROUPS = config.EXP2["a"]["groups"]


def _cfg(rule):
    return (config.INV_V1_BASELINE, True) if rule == "baseline" else (config.INV_SCALEFREE, False)


def run_inv(ev, d, N, nu, HN, rule):
    cfg, legacy = _cfg(rule)
    pr = Problem(ev, d, N, nu, GRID, cfg=cfg, legacy_baseline=legacy)
    g, info = invert(pr)
    return dict(eps_model=info["eps_model"], W1=w_to(g, GRID, *HN, 1), W2=w_to(g, GRID, *HN, 2),
                kappa_W1=kappa_hat(pr, g), success=info["status"] != "failed", status=info["status"],
                p0_emp=pr.p0_emp, p0_model=pr.p0_model)


def eig(X, n):
    return sampling.spectrum(X, n, check=True, clip=False)             # historical: unclipped eigvalsh


def beta_for_r2(s, target=2.0):
    lo, hi = 0.0, 100.0
    f = lambda b: np.sum(np.log1p(b * s) - 0.5 * np.log1p(2 * b * s)) - np.log(target)   # noqa: E731
    for _ in range(200):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def task_a(rep, group, rule):
    P = config.EXP2["a"]
    N, d, d0 = P["N"], P["d"], P["d0"]
    rng = config.rng("exp2a", rep, GROUPS.index(group))
    sig = np.where(np.arange(d) < d0, 0.5, 2.5)
    HN = (config.H0_ATOMS, config.H0_WEIGHTS)
    rows = []

    def gamma_lam():
        return sampling.gamma_weights(rng, 1.0, N)

    if group == "base":
        X = np.sqrt(sig)[:, None] * rng.standard_normal((d, N)); lam = gamma_lam()
        n_m = sampling.multinomial(rng, lam)
        n_r = sampling.residual(rng, lam)
        ev_m, ev_r = eig(X, n_m), eig(X, n_r)
        rows.append({"setting": "S1 multi->multi", **run_inv(ev_m, d, N, laws.plugin_multi(lam), HN, rule)})
        rows.append({"setting": "S2 res->res", **run_inv(ev_r, d, N, laws.plugin_res(lam), HN, rule)})
        rows.append({"setting": "S3 multi->res (wrong mechanism)", **run_inv(ev_m, d, N, laws.plugin_res(lam), HN, rule)})
    elif group.startswith("endo"):
        q = int(group[4:])
        X = np.sqrt(sig)[:, None] * rng.standard_normal((d, N))
        B, _ = np.linalg.qr(rng.standard_normal((d, q)))
        s = np.linalg.eigvalsh(B.T @ (sig[:, None] * B)); beta = beta_for_r2(s)
        y = B.T @ X
        lw = -0.5 * beta * np.sum(y * y, axis=0); lam = np.exp(lw - lw.max())
        lam *= N / lam.sum()  # I9: definitional -- endogenous weights self-normalised to mean 1 by design
        n = rng.multinomial(N, lam / N)
        rows.append({"setting": f"S4 endogenous q={q}", "r2N": float(np.mean(lam ** 2)),
                     **run_inv(eig(X, n), d, N, laws.plugin_multi(lam), HN, rule)})
        lamp = rng.permutation(lam); n2 = rng.multinomial(N, lamp / N)
        rows.append({"setting": f"S4c permuted control q={q}", "r2N": float(np.mean(lamp ** 2)),
                     **run_inv(eig(X, n2), d, N, laws.plugin_multi(lamp), HN, rule)})
    elif group.startswith("corr"):
        rho = float(group[4:])
        Y = np.sqrt(sig)[:, None] * rng.standard_normal((d, N))
        par = rng.integers(0, N, N)
        X = rho * Y[:, par] + np.sqrt(1 - rho ** 2) * np.sqrt(sig)[:, None] * rng.standard_normal((d, N))
        lam = gamma_lam(); n = rng.multinomial(N, lam / N)
        rows.append({"setting": f"S5 correlated particles rho={rho}",
                     **run_inv(eig(X, n), d, N, laws.plugin_multi(lam), HN, rule)})
    for r in rows:
        r.update(part="a", rep=rep)
    return rows


def task_b(c, al, rep, rule):
    N = config.EXP2["b"]["N"]; d = int(round(c * N))
    rng = config.rng("exp2b", int(c * 100), int(al * 10), rep)
    X = rng.standard_normal((d, N))
    lam = sampling.gamma_weights(rng, al, N)
    n = sampling.multinomial(rng, lam)
    res = run_inv(eig(X, n), d, N, laws.plugin_multi(lam), (np.array([1.0]), np.array([1.0])), rule)
    ps = 1 - (al / (al + 1)) ** al
    return [dict(part="b", c=c, alpha=al, rep=rep, p_s=ps, c_minus_ps=c - ps, **res)]


def dispatch(t):
    return task_a(*t[1:]) if t[0] == "a" else task_b(*t[1:])


def _keyed(pair):
    """(label, args) -> (label, rows), executed IN THE WORKER.

    See exp1._keyed: building `(label, dispatch(t))` in the parent would run the work in the PARENT
    and serialise the Pool.  Module-level so it is picklable."""
    label, args = pair
    return label, dispatch(args)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=("baseline", "m19"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default=None, help="'a:0-1' or 'b:0-1' (reps)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    Pa, Pb = config.EXP2["a"], config.EXP2["b"]
    ra, rb = range(Pa["R"]), range(Pb["R"])
    parts = ("a", "b")
    if a.subset:
        part, rr = a.subset.split(":"); lo, hi = (int(x) for x in rr.split("-"))
        parts = (part,)
        ra = rb = range(lo, hi + 1)
    tasks = []
    if "a" in parts:
        tasks += [("a", r, g, a.rule) for r in ra for g in GROUPS]
    if "b" in parts:
        tasks += [("b", c, al, r, a.rule) for c in Pb["cs"] for al in Pb["alphas"] for r in rb]
    os.makedirs(a.out, exist_ok=True)
    from multiprocessing import Pool
    from ..runbudget import drive, budget_status
    budget = config.run_budget("EXP-2")     # NOT `rb`: that name already holds the part-b rep range
    t0 = time.time(); rows = []
    with Pool(config.workers(a.workers)) as pool:
        # stage F: the run-level budget wraps the loop.  The stream is `pool.imap_unordered` over
        # (_label, args) pairs and `_keyed` runs IN THE WORKER, so the Pool's parallelism is exactly
        # what it was -- see drive()'s docstring for why building (label, dispatch(t)) here is wrong.
        pairs = pool.imap_unordered(_keyed, [(str(t), t) for t in tasks])
        for label, rr in drive(budget, pairs, a.out):
            rows += rr
            if (len(budget.completed) if budget else 0) % 20 == 0:
                print(f"{len(budget.completed) if budget else 0}/{len(tasks)} {time.time() - t0:.0f}s",
                      flush=True)
    with open(os.path.join(a.out, "exp2_rows.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    status = budget_status(budget)
    print(f"EXP-2 ({a.rule}) {len(tasks)} tasks, {time.time() - t0:.0f}s -> {a.out} [{status}]")
    return 2 if status == "budget_exceeded" else 0


if __name__ == "__main__":
    sys.exit(main())
