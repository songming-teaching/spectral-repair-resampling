r"""S7 (supplementary): the activation region of the supervisor's clause (b), Bhat > delta_fb.

Migrated onto the srlab core (P3) from archive/code_2026-09-27/s7_activation.py (the M14 / pre-M14
producer, `s7_activation_v3_M14.json`) and s7_M15_refit.py (the M15 producer,
`s7_activation_M15_refit.json`).  Same cells, seeds, grids and bandwidths as the archive: this is a
REPRODUCTION module, so every number below is fixed by the archive, not chosen here.

TWO MODES, EXPLICITLY SEPARATED (v44, a recorded design decision):
  * ``--mode archive``  CPU-PINNED bitwise reproduction of the archived rows.  The migration claim is
    CPU-vs-CPU: the archive was produced by numpy, and since the GPU path became the default
    (M40) ``forward.law_quantiles`` would route delta_fb to the GPU and land it at ~1e-11, far
    outside a bitwise assertion.  Everything therefore runs inside ``forward.pin_cpu_backend()``.
  * ``--mode window``   the GPU production path (M40), i.e. the numbers the paper will carry.  The
    GPU/CPU difference is expected and is registered under the v35 two-layer tolerance: integral
    layer (moments, W1, delta_fb) <= 1e-13 is a non-content change; the pointwise layer <= 1e-10 is
    a backend-change.  THIS MODE IS NOT RUN IN THE MIGRATION BATCH -- it is the window's job.

cfg DISCIPLINE (v44; the z-grid.txt finding):  the platform ``Problem`` default is
INV_TABLEI_STRONG = InversionCfg(n_x=400, h_rel=0.01), which is NOT the archive's 160 / h_rel.  A
module that inherited the default would silently compute a different z-grid -- the exact "silently
changing numbers" failure the recorded decision names.  Every Problem here therefore gets an EXPLICIT, COMPLETE
cfg from ``_cfg``; the archive (s7_activation.py:50) passed n_x=NX, h_rel=hr through **kw.
Signature change: the archive's ``Problem(..., n_x=NX, h_rel=hr)`` no longer exists, and no
compatibility shim is added -- the call is rewritten explicitly, as the recorded decision requires.

gamma: the archive used the SCALE-FREE rule (pr.gamma = 1e-8/mean(ev^2), s7_activation.py:51), which
is exactly config.INV_SCALEFREE / gamma_rule="scale_free".  S7 therefore has NO gamma switch (unlike
EXP-1/2/3, which migrate from the legacy rule) -- both modes run the same rule.

Usage:
  python -m srlab.experiments.s7 --mode archive --out results/s7_archive [--subset 0-1] [--workers 8]
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
from ..forward import pin_cpu_backend                                  # noqa: E402
from ..inverse import InversionCfg, Problem, invert, kappa_hat         # noqa: E402
from ..invariants import ConvergenceError, InvariantError, abstention_reason   # noqa: E402
from .. import supervisor                                              # noqa: E402


def _code(exc):
    """The bracketed invariant code of an InvariantError message ("[I3] ..."), or "". """
    msg = str(exc)
    return msg[1:msg.index("]")] if msg.startswith("[") and "]" in msg else ""

CS = config.S7["cs"]
ALPHAS = config.S7["alphas"]
D = config.S7["D"]
HRELS = config.S7["hrels"]
NX = config.S7["NX"]


def _cfg(h_rel):
    """The archive's inversion configuration, CONSTRUCTED IN FULL (v44: no inherited defaults).

    n_x / h_rel are the two fields that differ from the platform default (INV_TABLEI_STRONG, 400 /
    0.01); gamma_rule / gamma_rel reproduce s7_activation.py:51's scale-free rule."""
    return InversionCfg(n_x=NX, h_rel=h_rel, x_lo_rel=0.0, method="SLSQP", maxiter=500, ftol=1e-12,
                        init="uniform", objective_scaling="L0", restarts=1,
                        gamma_rule="scale_free", gamma_rel=1e-8)


def _draw(c, al, rep):
    """Cell data, draw order and grid -- verbatim from s7_activation.task (seed stream s7_cell)."""
    N = int(round(D / c))
    rng = config.rng("s7_cell", int(c * 100), int(al * 10), rep)
    X = rng.standard_normal((D, N))
    lam = sampling.gamma_weights(rng, al, N)
    n = sampling.multinomial(rng, lam)
    mr = X @ n / N
    ev = np.maximum(np.linalg.eigvalsh((X * n[None, :]) @ X.T / N - np.outer(mr, mr)), 0.0)
    nu = laws.plugin_multi(lam)
    grid = (np.round(np.exp(np.linspace(np.log(0.05), np.log(1.05 * np.quantile(ev, 0.999)))), 8)
            if np.quantile(ev, 0.999) > 2.6 else np.round(np.arange(0.2, 2.6001, 0.025), 10))
    return N, lam, n, ev, nu, grid


def _eta_rng(c, al, rep, hr):
    return config.rng("s7_eta", int(c * 100), int(al * 10), rep, int(hr * 1000))


def task(args):
    """One (c, alpha, rep) cell -> the M14 row and the M15 row, both matching the archived schema."""
    c, al, rep, mode = args
    N, lam, n, ev, nu, grid = _draw(c, al, rep)

    m14 = dict(c=c, alpha=al, rep=rep, N=N, p0_distortion=laws.zero_atom(float(nu[0]), D / N),
               p0_emp=float(np.mean(ev < 1e-9 * max(ev.max(), 1e-12))), by_h={})
    m15 = dict(c=c, alpha=al, rep=rep, by_h={})

    def run():
        legacy = (mode == "archive")      # a recorded design decision ①: the archive path had no I3 guard at all
        for hr in HRELS:
            cfg = _cfg(hr)
            pr = Problem(ev, D, N, nu, grid, cfg=cfg)
            try:
                g, info = invert(pr)
                kap = kappa_hat(pr, g)
                pr_r = Problem(ev, D, N, laws.realized(n), grid, zgrid=pr.z, cfg=cfg)
                gap = float(np.sqrt(np.mean(np.abs(pr_r.model(g) - pr.model(g)) ** 2)))
                d_fb, dfb_info = supervisor.delta_fb_W1(grid, g, n, D / N, legacy_unchecked=legacy,
                                                        return_info=True)
                Rg = float(np.dot(g, grid ** 2))
            except (ConvergenceError, InvariantError) as exc:
                # §12.5 / M21: a typed numerical event becomes a REGISTERED ABSTENTION, never an
                # aborted campaign.  The archive silently used the unconverged value here; the
                # platform refuses to, and the cell is marked so the recorded decision can see it.
                reason = abstention_reason(exc) if isinstance(exc, ConvergenceError) else "invariant:" + _code(exc)
                m14["by_h"][f"{hr}"] = dict(h_rel=hr, abstain=True, abstain_reason=reason,
                                            message=str(exc)[:300])
                m15["by_h"][f"{hr}"] = dict(abstain=True, abstain_reason=reason, message=str(exc)[:300])
                continue

            # --- M14 row (fixed-Hhat floor; pre-M14 Bhat is the un-de-floored one) -------------
            B = supervisor.budget_Bhat(kap, info["eps_model"], gap, pr.gamma, Rg)
            eta, eta_all = supervisor.eta_disc(pr, g, lam, rng=_eta_rng(c, al, rep, hr), B=3)
            B14, eps_t = supervisor.budget_Bhat_M14(kap, info["eps_model"], eta, gap, pr.gamma, Rg)
            m14["by_h"][f"{hr}"] = dict(
                h_rel=hr, kappa=kap, eps_model=info["eps_model"], gap=gap, delta_fb=d_fb,
                Bhat=B, ratio=B / max(d_fb, 1e-12), activated=bool(B > d_fb),
                success=info["status"] != "failed", eta_disc=eta, eta_disc_all=eta_all,
                eps_tilde=eps_t, Bhat_M14=B14, ratio_M14=B14 / max(d_fb, 1e-12),
                activated_M14=bool(B14 > d_fb),
                # O-3 / M40 boundary 1: the backend behind delta_fb reaches the row, hence the manifest
                path=dfb_info.get("path"), eta_ladder=dfb_info.get("eta_ladder"))

            # --- M15 row (per-sample refit floor; the paper cites M15) -------------------------
            eta_r, _ = supervisor.eta_disc(pr, g, lam, rng=_eta_rng(c, al, rep, hr), B=3, refit=True)
            B15, eps_t15 = supervisor.budget_Bhat_M14(kap, info["eps_model"], eta_r, gap, pr.gamma, Rg)
            m15["by_h"][f"{hr}"] = dict(
                eps_model=info["eps_model"], eta_refit=eta_r, eps_tilde=eps_t15, kappa=kap, gap=gap,
                delta_fb=d_fb, Bhat=B15, ratio=B15 / max(d_fb, 1e-12), activated=bool(B15 > d_fb),
                path=dfb_info.get("path"))

    if mode == "archive":
        # CPU vs CPU: the archive is a numpy product, so only a pinned CPU run can be bitwise.
        with pin_cpu_backend():
            run()
    else:
        run()
    return m14, m15


def _keyed(pair):
    """(label, args) -> (label, (m14, m15)), executed IN THE WORKER (see exp1._keyed)."""
    label, args = pair
    return label, task(args)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("archive", "window"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default=None, help="'first-last' over the cell list (for a pilot)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)

    tasks = [(c, al, r, a.mode) for c in CS for al in ALPHAS for r in range(3)]
    if a.subset:
        lo, hi = (int(x) for x in a.subset.split("-"))
        tasks = tasks[lo:hi + 1]
    os.makedirs(a.out, exist_ok=True)

    from multiprocessing import Pool
    from ..runbudget import drive, budget_status
    budget = config.run_budget("S7")
    t0 = time.time()
    rows14, rows15 = [], []
    with Pool(config.workers(a.workers)) as pool:
        pairs = pool.imap_unordered(_keyed, [(f"c{t[0]}|a{t[1]}|r{t[2]}", t) for t in tasks])
        for label, (m14, m15) in drive(budget, pairs, a.out):
            rows14.append(m14)
            rows15.append(m15)
            print(f"{len(rows14)}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)

    key = lambda r: (r["c"], r["alpha"], r["rep"])            # noqa: E731
    rows14.sort(key=key)
    rows15.sort(key=key)
    for name, rows in (("s7_activation_v3_M14.json", rows14),
                       ("s7_activation_M15_refit.json", rows15)):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
    status = budget_status(budget)
    print(f"S7 ({a.mode}) {len(tasks)} cells, {time.time() - t0:.0f}s -> {a.out} [{status}]")
    return 2 if status == "budget_exceeded" else 0


if __name__ == "__main__":
    sys.exit(main())
