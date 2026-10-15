r"""EXP-1: static spectral distortion and repair -- migrated onto the srlab core (P3).

Same design as archive/code_2026-09-27/exp1.py (the producer of experiment/data/exp1/exp1_rows.json):
H0 = 0.7 delta_0.5 + 0.3 delta_2.5, c = 0.75, N in {500, 1000, 2000}, alpha in {10, 1, 0.3}, R = 50,
two resampling schemes x {none, oracle, cond, realized} inverters.  Every random draw, in the same order,
comes from SeedSequence([MASTER, 1, N, int(10 alpha), rep]).

  --rule baseline   pre-M19 gamma0/N (frozen baseline INV_V1_BASELINE): must reproduce the archived
                    rows bitwise -- the migration check;
  --rule m19        the scale-free rule (M19) -- the window rerun.

Usage:  python -m srlab.experiments.exp1 --rule baseline --out results/exp1_baseline [--subset N:rep0-rep1] [--workers 8]
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
from ..stats import wasserstein                                        # noqa: E402

C = config.EXP1["c"]
GRID = config.GRID_005
H_ATOMS = config.H0_ATOMS


def task(args):
    N, al, rep, rule = args
    cfg = config.INV_V1_BASELINE if rule == "baseline" else config.INV_SCALEFREE
    rng = config.rng("exp1", N, int(al * 10), rep)
    d = int(round(C * N)); d0 = int(round(0.7 * d))
    sig = np.where(np.arange(d) < d0, 0.5, 2.5)
    HN = (H_ATOMS, np.array([d0 / d, 1 - d0 / d]))
    lam = sampling.gamma_weights(rng, al, N)
    X = np.sqrt(sig)[:, None] * rng.standard_normal((d, N))
    r2N = float(np.mean(lam ** 2))
    k = np.floor(lam).astype(int); RN = N - int(k.sum())
    counts = {"multi": sampling.multinomial(rng, lam), "res": sampling.residual(rng, lam)}
    rows = []
    for scheme in ("multi", "res"):
        n = counts[scheme]
        ev = sampling.spectrum(X, n, check=True, clip=False)           # historical: unclipped eigvalsh
        m2 = float(np.sum(ev ** 2) / d)
        pos, p0 = sampling.split_atom(ev)
        EU_N = RN / N
        m2_th = HN[1] @ HN[0] ** 2 + (d / N) * (HN[1] @ HN[0]) ** 2 * ((1 + r2N) if scheme == "multi" else (r2N + EU_N))
        base = dict(N=N, d=d, alpha=al, rep=rep, scheme=scheme, r2N=r2N, EU_N=EU_N, p0_emp=p0, m2_emp=m2,
                    m2_theory=float(m2_th), b2N=float(np.mean(n.astype(float) ** 2)))
        evc = np.maximum(ev, 0.0)
        wts = np.full(len(ev), 1.0 / len(ev))
        rows.append({**base, "inverter": "none", "W1": wasserstein(evc, wts, *HN, 1), "W2": wasserstein(evc, wts, *HN, 2)})
        if rule == "baseline":
            nu_or, or_m1_err = laws.oracle_legacy(scheme, al)     # D23: recorded, never silent
        else:
            nu_or, or_m1_err = laws.oracle(scheme, al), 0.0
        nus = {"oracle": nu_or,
               "cond": laws.plugin_multi(lam) if scheme == "multi" else laws.plugin_res(lam),
               "realized": laws.realized(n)}
        z = None
        for inv, nu in nus.items():
            pr = Problem(ev, d, N, nu, GRID, zgrid=z, cfg=cfg, legacy_baseline=(rule == "baseline"))
            z = pr.z
            t0 = time.time(); g, info = invert(pr); dt = time.time() - t0
            kap = kappa_hat(pr, g) if inv == "cond" else float("nan")
            rows.append({**base, "inverter": inv, "W1": w_to(g, GRID, *HN, 1), "W2": w_to(g, GRID, *HN, 2),
                         "eps_model": info["eps_model"], "kappa_W1": kap, "p0_model": pr.p0_model,
                         "nit": info["nit"], "success": info["status"] != "failed", "status": info["status"],
                         "attempts": info["attempts"], "sec": dt,
                         **({"nu_m1_error": or_m1_err} if inv == "oracle" else {})})
    return rows


def _keyed(pair):
    """(label, args) -> (label, rows), executed IN THE WORKER.

    The label must be computed where the work is: building `(label, task(t))` in the parent would
    call `task(t)` in the PARENT and serialise the Pool -- the exact mistake drive()'s docstring
    warns about.  A module-level function so it is picklable."""
    label, args = pair
    return label, task(args)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=("baseline", "m19"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default=None, help="e.g. '500:0-2' -> N=500, reps 0..2, all alpha")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    Ns, reps = config.EXP1["Ns"], range(config.EXP1["R"])
    if a.subset:
        n_s, r_s = a.subset.split(":")
        Ns = [int(n_s)]
        lo, hi = (int(x) for x in r_s.split("-"))
        reps = range(lo, hi + 1)
    tasks = [(N, al, r, a.rule) for N in Ns for al in config.EXP1["alphas"] for r in reps]
    os.makedirs(a.out, exist_ok=True)
    from multiprocessing import Pool
    from ..runbudget import drive, budget_status
    rb = config.run_budget("EXP-1")
    t0 = time.time(); allrows = []
    with Pool(config.workers(a.workers)) as pool:
        # stage F: the run-level budget wraps the loop.  The stream is `pool.imap_unordered` over
        # (_label, args) pairs and `_keyed` runs IN THE WORKER, so the Pool's parallelism is exactly
        # what it was -- see drive()'s docstring for why building (label, task(t)) here would be wrong.
        pairs = pool.imap_unordered(_keyed, [(f"N{t[0]}|a{t[1]}|rep{t[2]}", t) for t in tasks])
        for label, rows in drive(rb, pairs, a.out):
            allrows += rows
            if len(allrows) % 10 == 0:
                print(f"{len(rb.completed) if rb else 0}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    allrows.sort(key=lambda r: (r["N"], r["alpha"], r["rep"], r["scheme"], r["inverter"]))
    with open(os.path.join(a.out, "exp1_rows.json"), "w", encoding="utf-8") as fh:
        json.dump(allrows, fh)
    status = budget_status(rb)
    print(f"EXP-1 ({a.rule}) {len(tasks)} tasks, {time.time() - t0:.0f}s -> {a.out} [{status}]")
    return 2 if status == "budget_exceeded" else 0


if __name__ == "__main__":
    sys.exit(main())
