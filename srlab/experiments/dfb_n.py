r"""S4(c) / M24: delta_fb versus N at FIXED aspect ratio c = 0.75 (EXPERIMENT_PLAN_SUPPLEMENT §1).

Why a new module and not S7: in S7, N = D/c with D fixed, so its six N are six different c -- a sweep
over N there changes the PROBLEM, not the sample size.  Here c is held at 0.75 and d = round(c N)
grows with N, so the N-dependence of delta_fb is a sample-size effect.

Design (registered in PREREGISTRATION M24, execution record 2026-10-05):
  H   = H0 = 0.7 delta_0.5 + 0.3 delta_2.5   (EXP-1's population; c = 0.75 is EXP-1's c)
  main slice      N in {500, 1000, 2000, 4000} x alpha in {10.0, 0.3}
  auxiliary slice N in {500, 2000}             x alpha in {3.0, 1.0}   (cell-aligned with probe_i6)
  R = 20 reps, multinomial resampling, conditional plug-in law (laws.plugin_multi), grid GRID_005,
  scale-free gamma (INV_SCALEFREE), h_rel in {0.03 (deployed), 0.01, 0.003}.
Every row records the M26 cap fields of the delta_fb quantile call; a typed numerical event is a
REGISTERED abstention (M21), never a silent value and never an aborted campaign.

Usage:  python -m srlab.experiments.dfb_n --out results/window/dfb_N [--subset N:alpha] [--reps 0-19]
"""
import argparse
import dataclasses
import json
import os
import sys
import time

from ..threads import pin as _pin
_pin("default")                     # D25/M37: forced before numpy loads

import numpy as np                                                     # noqa: E402

from .. import config, laws, sampling                                  # noqa: E402
from ..inverse import Problem, invert                                  # noqa: E402
from ..invariants import ConvergenceError, InvariantError, abstention_reason   # noqa: E402
from .. import supervisor                                              # noqa: E402

P = config.DFBN
GRID = config.GRID_005


def cells():
    main = [(N, al) for N in P["Ns"] for al in P["alphas_main"]]
    aux = [(N, al) for N in P["Ns_aux"] for al in P["alphas_aux"]]
    return main + aux


def _code(exc):
    msg = str(exc)
    return msg[1:msg.index("]")] if msg.startswith("[") and "]" in msg else ""


def task(args):
    N, al, rep = args
    c = P["c"]
    rng = config.rng("dfbN", N, int(al * 10), rep)
    d = int(round(c * N)); d0 = int(round(0.7 * d))
    sig = np.where(np.arange(d) < d0, 0.5, 2.5)
    lam = sampling.gamma_weights(rng, al, N)
    X = np.sqrt(sig)[:, None] * rng.standard_normal((d, N))
    n = sampling.multinomial(rng, lam)
    ev = sampling.spectrum(X, n, check=True, clip=True)
    nu = laws.plugin_multi(lam)
    nu_r = laws.realized(n)
    row = dict(N=N, d=d, c=d / N, alpha=al, rep=rep,
               p0_emp=float(np.mean(ev <= 1e-9 * ev.max())), n_active=int(np.count_nonzero(n)), by_h={})
    for hr in P["hrels"]:
        cfg = dataclasses.replace(config.INV_SCALEFREE, h_rel=hr)
        t0 = time.time()
        try:
            pr = Problem(ev, d, N, nu, GRID, cfg=cfg)
            g, info = invert(pr)
            t1 = time.time()
            dfb, di = supervisor.delta_fb_W1(GRID, g, n, d / N, nu_realized=nu_r, return_info=True)
            t2 = time.time()
        except (ConvergenceError, InvariantError) as exc:
            reason = abstention_reason(exc) if isinstance(exc, ConvergenceError) else "invariant:" + _code(exc)
            row["by_h"][f"{hr}"] = dict(h_rel=hr, abstain=True, abstain_reason=reason,
                                        info=getattr(exc, "info", None), message=str(exc)[:300],
                                        sec=time.time() - t0)
            continue
        cap = di.get("cap") or {}
        row["by_h"][f"{hr}"] = dict(
            h_rel=hr, abstain=False, delta_fb=dfb, eps_model=info["eps_model"], status=info["status"],
            sec_invert=t1 - t0, sec_dfb=t2 - t1,
            cap=cap, n_grid=di.get("n_grid"), eta_ladder=di.get("eta_ladder"),
            path=di.get("path"), paths_seen=di.get("paths_seen"), max_resid=di.get("max_resid"),
            returned_unconverged=di.get("returned_unconverged"),
            worst_unconverged_resid=di.get("worst_unconverged_resid"))
    return row


def _keyed(pair):
    label, args = pair
    return label, task(args)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default=None, help="'N:alpha' -> one cell (pilot)")
    ap.add_argument("--reps", default=None, help="'lo-hi' (inclusive)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    cl = cells()
    if a.subset:
        n_s, a_s = a.subset.split(":")
        cl = [(int(n_s), float(a_s))]
    reps = range(P["R"])
    if a.reps:
        lo, hi = (int(x) for x in a.reps.split("-"))
        reps = range(lo, hi + 1)
    # largest cells first so the pool's tail is short
    tasks = sorted([(N, al, r) for (N, al) in cl for r in reps], key=lambda t: (-t[0], t[1], t[2]))
    os.makedirs(a.out, exist_ok=True)

    from multiprocessing import Pool
    t0 = time.time(); rows = []
    part = os.path.join(a.out, "dfb_N_rows.partial.jsonl")
    with Pool(config.workers(a.workers)) as pool, open(part, "w", encoding="utf-8") as fp:
        for label, row in pool.imap_unordered(_keyed, [(f"N{t[0]}|a{t[1]}|rep{t[2]}", t) for t in tasks]):
            rows.append(row)
            fp.write(json.dumps(row) + "\n"); fp.flush()       # a crash keeps what was computed
            print(f"{len(rows)}/{len(tasks)} {label} {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["N"], r["alpha"], r["rep"]))
    with open(os.path.join(a.out, "dfb_N_rows.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    os.remove(part)
    n_abs = sum(1 for r in rows for v in r["by_h"].values() if v.get("abstain"))
    print(f"dfb_N {len(tasks)} tasks, {n_abs} abstained (cell x h), {time.time() - t0:.0f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
