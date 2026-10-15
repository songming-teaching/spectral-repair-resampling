r"""EXP-3: stable scalar cycle, supervisor, forced-abstention contrast -- migrated onto srlab (P3).

Design, draw order and seed streams as archive/code_2026-09-27/exp3.py:
  X_k = diag(sqrt(s_k)) Z_k;  s_{k+1} = a^2 sort(eig C_k) + q  with C_k = mu_hat_k (fallback) or
  lift(H_hat_k) (corrected); all methods share Z_k, lambda_k and the counts (paired).
  streams: exp3_main [MASTER, 4, rep, k], exp3_calib [MASTER, 5, rep, k], exp3_eta [MASTER, 11, rep, k].
  Supervisor: comparison rule, M14 floor (eta_disc at fixed Hhat, B = 3), clause (a) threshold
  tau = 2 x median eps_model over the calibration runs (cond method, stream 5, 10 reps).

Two rules:
  --rule baseline  frozen pre-M19 gamma and the HISTORICAL inversion contract (one attempt, an
                   unconverged g is used -- invert_baseline); reproduces the archive bitwise:
                   non-sup methods vs experiment/data/exp3/exp3_rows.json, sup vs the M17 replay
                   experiment/data/m17/exp3_check/exp3_sup_rows_fastdfb.json (exp3_rows.json's sup
                   column predates M14 and the M17 fix of delta_fb);
  --rule m19       scale-free gamma, M34 oracle law, and the M21 state machine: a solver failure is a
                   typed event mapped to a registered abstention with reason code
                   'solver_nonconvergence'; supervisor abstentions carry 'supervisor_clause_a/b';
                   forced abstentions carry 'forced_schedule'.  Every row records `path`.

Usage: python -m srlab.experiments.exp3 --rule baseline --out DIR [--methods sup,cond] [--reps 0-1] [--workers 1]
"""
import argparse
import json
import os
import sys
import time

from ..threads import pin as _pin
_pin("default")                     # D25/M37: forced before numpy loads

import numpy as np                                                         # noqa: E402

from .. import config, laws, sampling                                      # noqa: E402
from ..inverse import (Problem, invert_baseline, invert_or_abstain,        # noqa: E402
                       kappa_hat, lift_eigs)
from ..stats import wasserstein                                            # noqa: E402
from ..supervisor import delta_fb_W1, budget_Bhat, budget_Bhat_M14, eta_disc   # noqa: E402
from ..invariants import ConvergenceError, abstention_reason             # noqa: E402

P = config.EXP3
GRID = config.GRID_0025
d, N, a, q, K = P["d"], P["N"], P["a"], P["q"], P["K"]
SHOCKS = set(P["shocks"])
FORCED = {"forced_end": set(P["forced_end"]), "forced_uniform": set(P["forced_uniform"])}
H0x, H0w = config.H0_ATOMS, config.H0_WEIGHTS


def alpha_k(k):
    return P["alpha_shock"] if k in SHOCKS else P["alpha"]


def step_draws(stream, rep, k):
    rng = config.rng(stream, rep, k)
    Z = rng.standard_normal((d, N))
    al = alpha_k(k)
    lam = sampling.gamma_weights(rng, al, N)
    n = sampling.multinomial(rng, lam)
    return Z, lam, n, al


def _oracle(al, rule):
    return laws.oracle_legacy("multi", al)[0] if rule == "baseline" else laws.oracle("multi", al)


def run(method, rep, stream, thr, rule):
    cfg = config.INV_V1_BASELINE if rule == "baseline" else config.INV_SCALEFREE
    legacy = rule == "baseline"
    s = np.where(np.arange(d) < 525, 0.5, 2.5).astype(float)
    Hs = H0x.copy()
    uni = np.full(d, 1.0 / d)
    rows = []
    for k in range(K + 1):
        Z, lam, n, al = step_draws(stream, rep, k)
        X = np.sqrt(s)[:, None] * Z
        ev = sampling.spectrum(X, n, check=True, clip=True)                # historical max(eig, 0)
        row = dict(method=method, rep=rep, k=k, alpha=al, r2N=float(np.mean(lam ** 2)))
        # M42 / a recorded design decision: the SPECTRAL INSTRUMENTS, read off the counts for free (the count-rank
        # identity, I2) -- no eigendecomposition, no extra inversion.  n_active particles carry a
        # nonzero count, so the centred resampled covariance has rank <= n_active - 1: the zero atom
        # is max(0, 1 - (n_active-1)/d) and the EFFECTIVE DIMENSION is that rank.  These are
        # diagnostics (nothing downstream reads them), so adding them changes no existing number.
        n_active = int(np.count_nonzero(n))
        row["n_active"] = n_active
        row["p0_count_rank"] = max(0.0, 1.0 - (n_active - 1) / d)
        row["d_eff"] = n_active - 1
        W = lambda x, w, p: wasserstein(x, w, Hs, H0w, p)                    # noqa: E731
        row["delta_fb_W1"] = wasserstein(ev, uni, s, uni, 1)
        row["delta_fb_W2"] = wasserstein(ev, uni, s, uni, 2)
        # a recorded design decision escalation: the SAME-STATE fallback error.  Clause (b) never compares two
        # trajectories -- at step k the supervisor sees ONE state Hs_k and asks "correction design vs
        # fallback design, both against this same Hs_k".  `eta_W1` (below) is the correction design's
        # error at this state; this is the fallback design's error at the SAME state, so
        #     truth (should abstain)  <=>  eta_W1 > eta_fb_same_state
        # is the supervisor's own question measured instead of estimated.  It is a diagnostic column:
        # nothing downstream reads it, so the trajectory and every existing number are unchanged.
        row["eta_fb_same_state"] = W(ev, uni, 1)
        if method == "none":
            C = ev; row["abstain"] = True; row["path"] = "fallback:no_correction"
            row["eta_W1"], row["eta_W2"] = W(ev, uni, 1), W(ev, uni, 2)
            rows.append(row)
            s = a * a * np.sort(C) + q; Hs = a * a * Hs + q
            continue
        nu = _oracle(al, rule) if method == "oracle" else laws.plugin_multi(lam)
        pr = Problem(ev, d, N, nu, GRID, cfg=cfg, legacy_baseline=legacy)
        if legacy:
            g, info = invert_baseline(pr); failed = False
            row["success"] = info["success"]
        else:
            g, rec = invert_or_abstain(pr)
            failed = rec["abstain"]
            info = rec["info"] if not failed else rec["detail"]
            row["solver_status"] = "typed_failure" if failed else info["status"]
        abstain, reason = False, None
        if not failed:
            kap = kappa_hat(pr, g)
            row.update(eps_model=info["eps_model"], kappa_W1=kap,
                       b_W1=wasserstein(GRID, g, s, uni, 1), b_W2=wasserstein(GRID, g, s, uni, 2))
            if method in FORCED and k in FORCED[method]:
                abstain, reason = True, "forced_schedule"
            if method == "sup":
                nu_r = laws.realized(n)
                pr_r = Problem(ev, d, N, nu_r, GRID, cfg=cfg, legacy_baseline=legacy)
                gap = float(np.sqrt(np.mean(np.abs(pr_r.model(g) - pr.model(g)) ** 2)))
                try:
                    d_fb, dfb_info = delta_fb_W1(GRID, g, n, d / N, nu_realized=nu_r, return_info=True,
                                                 legacy_unchecked=legacy)
                except ConvergenceError as exc:              # O-1(c)/M38: catch the BASE, not one subclass
                    # M35 designed this for SolverEscalationFailure, but TailBudgetExceeded is its
                    # SIBLING and a bare ConvergenceError is its PARENT -- a narrow `except` let two
                    # of the three escape and abort the campaign.  Any convergence event here is a
                    # registered abstention; the mechanism is named by the exception's reason code.
                    d_fb, dfb_info = None, dict(typed_failure=str(exc), reason_code=abstention_reason(exc),
                                                **getattr(exc, "info", {}))
                Rg = float(np.dot(g, GRID ** 2))
                if d_fb is None:
                    row.update(delta_fb_hat=None, dfb_info=dfb_info, gap_realized=gap)
                    row["abstain"] = True; row["abstain_reason"] = "dfb_solver_nonconvergence"
                    C = ev; row["eta_W1"], row["eta_W2"] = W(ev, uni, 1), W(ev, uni, 2)
                    row["path"] = "fallback:dfb_solver_nonconvergence"
                    rows.append(row)
                    s = a * a * np.sort(C) + q; Hs = a * a * Hs + q
                    continue
                B_old = budget_Bhat(kap, info["eps_model"], gap, pr.gamma, Rg)
                eta, _ = eta_disc(pr, g, lam, rng=config.rng("exp3_eta", rep, k), B=P["eta_B"])
                B_hat, eps_t = budget_Bhat_M14(kap, info["eps_model"], eta, gap, pr.gamma, Rg)
                row.update(delta_fb_hat=d_fb, dfb_info=dfb_info, B_hat=B_hat, B_hat_preM14=B_old, eta_disc=eta,
                           eps_tilde=eps_t, gap_realized=gap)
                if info["eps_model"] > thr["tau"]:
                    abstain, reason = True, "supervisor_clause_a"
                elif B_hat > d_fb:
                    abstain, reason = True, "supervisor_clause_b"
        else:
            abstain, reason = True, "solver_nonconvergence"                  # M21: registered, never an abort
            if method in FORCED and k in FORCED[method]:
                reason = "forced_schedule+solver_nonconvergence"
        row["abstain"] = bool(abstain)
        if not legacy:
            row["abstain_reason"] = reason
        if abstain:
            C = ev; row["eta_W1"], row["eta_W2"] = W(ev, uni, 1), W(ev, uni, 2)
            row["path"] = f"fallback:{reason}"
        else:
            C = lift_eigs(g, GRID, d); row["eta_W1"], row["eta_W2"] = W(GRID, g, 1), W(GRID, g, 2)
            row["path"] = "corrected"
        rows.append(row)
        s = a * a * np.sort(C) + q
        Hs = a * a * Hs + q
    return rows


def job(args):
    return run(*args)


def _keyed(pair):
    """(label, args) -> (label, rows), executed IN THE WORKER.

    See exp1._keyed: building `(label, job(t))` in the parent would run the work in the PARENT and
    serialise the Pool.  Module-level so it is picklable."""
    label, args = pair
    return label, job(args)


def calibrate(pool, rule, reps=None):
    reps = range(P["R_calib"]) if reps is None else reps
    cal = sum(pool.map(job, [("cond", r, "exp3_calib", None, rule) for r in reps]), [])
    thr = dict(tau=2.0 * float(np.median([r["eps_model"] for r in cal if "eps_model" in r])),
               rule="v6: abstain iff eps_model > tau or Bhat > delta_fb")
    return thr, cal


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=("baseline", "m19"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default=",".join(P["methods"]))
    ap.add_argument("--reps", default=None, help="e.g. '0-1'")
    ap.add_argument("--tau-from", default=None, help="baseline check: take tau from this calibration json instead of recomputing")
    ap.add_argument("--workers", type=int, default=None)
    a_ = ap.parse_args(argv)
    reps = range(P["R"])
    if a_.reps:
        lo, hi = (int(x) for x in a_.reps.split("-")); reps = range(lo, hi + 1)
    os.makedirs(a_.out, exist_ok=True)
    from multiprocessing import Pool
    from ..runbudget import drive, budget_status
    rb = config.run_budget("EXP-3")
    t0 = time.time()
    with Pool(config.workers(a_.workers)) as pool:
        if a_.tau_from:
            thr = json.load(open(a_.tau_from, encoding="utf-8"))["threshold"]; cal = None
        else:
            thr, cal = calibrate(pool, a_.rule)
        with open(os.path.join(a_.out, "exp3_calibration.json"), "w", encoding="utf-8") as fh:
            json.dump({"threshold": thr, "calibration_rows": cal}, fh)
        tasks = [(m, r, "exp3_main", thr, a_.rule) for m in a_.methods.split(",") for r in reps]
        rows = []
        # stage F: the run-level budget wraps the loop.  The stream is `pool.imap_unordered` over
        # (_label, args) pairs and `_keyed` runs IN THE WORKER, so the Pool's parallelism is exactly
        # what it was -- see drive()'s docstring for why building (label, job(t)) here is wrong.
        pairs = pool.imap_unordered(_keyed, [(f"{t[0]}|rep{t[1]}", t) for t in tasks])
        for label, rr in drive(rb, pairs, a_.out):
            rows += rr
            print(f"{len(rb.completed) if rb else 0}/{len(tasks)} {time.time() - t0:.0f}s",
                  flush=True)
    with open(os.path.join(a_.out, "exp3_rows.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    status = budget_status(rb)
    print(f"EXP-3 ({a_.rule}) {len(tasks)} runs, {time.time() - t0:.0f}s -> {a_.out} [{status}]")
    return 2 if status == "budget_exceeded" else 0


if __name__ == "__main__":
    sys.exit(main())
