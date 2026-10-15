"""srlab.solver -- the separable fixed point behind G boxtimes T_c(nu).

    A(e) = sum_k p_k t_k / (1 + c t_k e),   D = tau A(e) - z,
    e    = sum_j g_j tau_j / D_j,            m(z) = sum_j g_j / D_j.

Two kernels share one Newton update:
  * `solve_e`      -- whole-array iteration (the historical arithmetic, kept verbatim so that the
                      inversion path reproduces bit-for-bit); used on small z-grids (<= a few hundred).
  * `newton_active`-- per-point active set; used on dense grids (M18: equal to the reference to 1e-15).
Every public entry point verifies the charter on its RETURNED values: per-point residual (I5) and
branch (I4).  Diagnostics are returned, never stored in module state.
"""
import numpy as np
from .config import SOLVER
from .invariants import check_branch, check_converged, SolverEscalationFailure


def solve_e(z, tau, g, t, p, c, e0=None, maxit=SOLVER.maxit, tol=SOLVER.tol):
    """Historical whole-array Newton iteration (arithmetic identical to archive srcore.solve_e).
    Returns (e, A, D, resid, iterations).  Does NOT check the charter -- callers do."""
    z = np.asarray(z, complex)
    tau = np.asarray(tau, float); g = np.asarray(g, float)
    t = np.asarray(t, float); p = np.asarray(p, float)
    a1 = float(np.dot(g, tau))
    e = (-a1 / z) if e0 is None else np.array(e0, complex)
    it = 0
    for it in range(1, maxit + 1):
        A = (p[None, :] * t[None, :] / (1.0 + c * t[None, :] * e[:, None])).sum(1)
        D = tau[None, :] * A[:, None] - z[:, None]
        F = (g[None, :] * tau[None, :] / D).sum(1)
        Ap = -c * (p[None, :] * t[None, :] ** 2 / (1.0 + c * t[None, :] * e[:, None]) ** 2).sum(1)
        dF = -(g[None, :] * tau[None, :] ** 2 / D ** 2).sum(1) * Ap
        R = e - F
        step = R / (1.0 - dF)
        enew = e - step
        bad = (enew.imag <= 0) | ~np.isfinite(enew)
        enew[bad] = 0.5 * e[bad] + 0.5 * F[bad]
        if np.max(np.abs(enew - e)) < tol * (1.0 + np.max(np.abs(e))):
            e = enew
            break
        e = enew
    A = (p[None, :] * t[None, :] / (1.0 + c * t[None, :] * e[:, None])).sum(1)
    D = tau[None, :] * A[:, None] - z[:, None]
    resid = np.abs(e - (g[None, :] * tau[None, :] / D).sum(1))
    return e, A, D, resid, it


def solve_e_pointwise(z, tau, g, t, p, c, e0=None, maxit=SOLVER.maxit, tol=SOLVER.tol):
    """M35: the SAME Newton update as solve_e, with a POINTWISE stopping rule -- each point leaves the
    active set once |e_new - e| < tol (1 + |e|) at that point.  Replaces the historical global rule
    (stop when the MAXIMUM step is small), which (i) made every point iterate as long as the slowest
    one (the audit's cost source) and (ii) could stop while an individual point had not converged
    (this session's silent source).  Returns (e, A, D, resid, max iterations)."""
    z = np.asarray(z, complex)
    tau = np.asarray(tau, float); g = np.asarray(g, float)
    t = np.asarray(t, float); p = np.asarray(p, float)
    a1 = float(np.dot(g, tau))
    e = (-a1 / z) if e0 is None else np.array(e0, complex)
    act = np.arange(len(z)); it = 0
    for it in range(1, maxit + 1):
        ea, za = e[act], z[act]
        den = 1.0 + c * t[None, :] * ea[:, None]
        A_ = (p[None, :] * t[None, :] / den).sum(1)
        D_ = tau[None, :] * A_[:, None] - za[:, None]
        F = (g[None, :] * tau[None, :] / D_).sum(1)
        Ap = -c * (p[None, :] * t[None, :] ** 2 / den ** 2).sum(1)
        dF = -(g[None, :] * tau[None, :] ** 2 / D_ ** 2).sum(1) * Ap
        en = ea - (ea - F) / (1.0 - dF)
        bad = (en.imag <= 0) | ~np.isfinite(en)
        en[bad] = 0.5 * ea[bad] + 0.5 * F[bad]
        done = np.abs(en - ea) < tol * (1.0 + np.abs(ea))
        e[act] = en
        act = act[~done]
        if act.size == 0:
            break
    A = (p[None, :] * t[None, :] / (1.0 + c * t[None, :] * e[:, None])).sum(1)
    D = tau[None, :] * A[:, None] - z[:, None]
    resid = np.abs(e - (g[None, :] * tau[None, :] / D).sum(1))
    return e, A, D, resid, it


ESCALATION = (
    # depth 1 -- per-point upgrade: dense continuation, 17 levels 256 -> 1 by factors sqrt(2)
    dict(levels=tuple(2.0 ** (8 - 0.5 * j) for j in range(17)), maxit=SOLVER.maxit),
    # depth 2 -- single-point cap upgrade: 10 x iteration cap, continuation 4x denser again
    dict(levels=tuple(2.0 ** (8 - 0.25 * j) for j in range(33)), maxit=10 * SOLVER.maxit),
)


def _continuation_pointwise(z, tau, g, t, p, c, levels, maxit):
    e = None
    for sc in levels:
        e, *_ = solve_e_pointwise(z.real + 1j * z.imag * sc, tau, g, t, p, c, e0=e, maxit=maxit)
    return solve_e_pointwise(z, tau, g, t, p, c, e0=e, maxit=maxit)


def stieltjes_strict(z, tau, g, t, p, c, accept=SOLVER.accept_resid):
    """M35 strict solve: pointwise criterion, then the escalation sequence on the failing points only.
    Every returned point is charter-checked (I5, I4).  Raises SolverEscalationFailure (typed) if the
    sequence is exhausted.  Returns (m, e, A, D, info) with info['escalation'] = points ending at
    depth 0 / 1 / 2."""
    z = np.asarray(z, complex)
    e, A, D, resid, _ = _continuation_pointwise(z, tau, g, t, p, c, SOLVER.continuation, SOLVER.maxit)
    depth = np.zeros(len(z), int)
    for d_, spec in enumerate(ESCALATION, start=1):
        fail = np.nonzero(~(resid < accept))[0]
        if fail.size == 0:
            break
        e_f, _A, _D, r_f, _ = _continuation_pointwise(z[fail], tau, g, t, p, c, spec["levels"], spec["maxit"])
        e[fail] = e_f; depth[fail] = d_
        A = (p[None, :] * np.asarray(t)[None, :] / (1.0 + c * np.asarray(t)[None, :] * e[:, None])).sum(1)
        D = np.asarray(tau)[None, :] * A[:, None] - z[:, None]
        resid = np.abs(e - (np.asarray(g)[None, :] * np.asarray(tau)[None, :] / D).sum(1))
    esc = dict(depth0=int(np.sum(depth == 0)), depth1=int(np.sum(depth == 1)), depth2=int(np.sum(depth == 2)),
               points=len(z))
    fail = ~(resid < accept)
    if np.any(fail):
        esc.update(unconverged_after_escalation=int(fail.sum()), worst_resid=float(resid[fail].max()))
        raise SolverEscalationFailure(
            f"[I5/M35] {int(fail.sum())}/{len(z)} points unconverged after the full escalation "
            f"(worst resid {float(resid[fail].max()):.3e})", dict(escalation=esc))
    m = (np.asarray(g, float)[None, :] / D).sum(1)
    check_branch(e, m, "stieltjes_strict")
    return m, e, A, D, dict(escalation=esc, max_resid=float(resid.max()))


def stieltjes(z, tau, g, t, p, c, accept=SOLVER.accept_resid):
    """m(z) with continuation in Im z (historical schedule), charter-checked.
    Returns (m, e, A, D, info)."""
    z = np.asarray(z, complex)
    e = None
    its = []
    for s in SOLVER.continuation:
        e, A, D, _, it = solve_e(z.real + 1j * z.imag * s, tau, g, t, p, c, e0=e)
        its.append(it)
    e, A, D, resid, it = solve_e(z, tau, g, t, p, c, e0=e)
    its.append(it)
    m = (np.asarray(g, float)[None, :] / D).sum(1)
    check_converged(resid, accept, "stieltjes")
    check_branch(e, m, "stieltjes")
    return m, e, A, D, dict(iterations=its, max_resid=float(resid.max()))


def newton_active(z, tau, g, t, p, c, e, maxit=SOLVER.warm_maxit, tol=SOLVER.tol):
    """Per-point Newton (same update as solve_e); converged points leave the active set.
    Returns (e, indices still unconverged)."""
    e = e.copy(); act = np.arange(len(z))
    for _ in range(maxit):
        ea, za = e[act], z[act]
        den = 1.0 + c * t[None, :] * ea[:, None]
        A = (p * t / den).sum(1)
        Ap = -c * (p * t ** 2 / den ** 2).sum(1)
        D = tau[None, :] * A[:, None] - za[:, None]
        F = (g * tau / D).sum(1)
        dF = -(g * tau ** 2 / D ** 2).sum(1) * Ap
        en = ea - (ea - F) / (1.0 - dF)
        bad = (en.imag <= 0) | ~np.isfinite(en)
        en[bad] = 0.5 * ea[bad] + 0.5 * F[bad]
        done = np.abs(en - ea) < tol * (1.0 + np.abs(ea))
        e[act] = en; act = act[~done]
        if act.size == 0:
            break
    return e, act


def _resid(z, tau, g, t, p, c, e):
    A = (p * t / (1.0 + c * t[None, :] * e[:, None])).sum(1)
    D = tau[None, :] * A[:, None] - z[:, None]
    return np.abs(e - (g * tau / D).sum(1)), A, D


def stieltjes_dense(z, tau, g, t, p, c, n_coarse, block, accept=SOLVER.accept_resid, legacy_unchecked=False):
    """m(z) on a dense grid parallel to the real axis (constant Im z, increasing Re z).
    Coarse points by `stieltjes` (continuation), interpolated warm start, active-set Newton per
    block; unconverged fine points are re-solved by `stieltjes` and COUNTED in info (never silent);
    unconverged COARSE points (initial guesses only) are counted too, their fine values are checked.
    Zero-weight atoms are dropped (exact).  Returns (m, info)."""
    z = np.asarray(z, complex)
    x = z.real
    assert np.all(np.diff(x) > 0) and np.ptp(z.imag) == 0, "dense grid must be sorted with constant Im z"
    tau = np.asarray(tau, float); g = np.asarray(g, float)
    t = np.asarray(t, float); p = np.asarray(p, float)
    keep = g > 0
    tau, g = tau[keep], g[keep]
    stride = max(1, len(z) // n_coarse)
    ic = np.unique(np.r_[np.arange(0, len(z), stride), len(z) - 1])
    # Coarse stage = the historical continuation solve, used ONLY as the initial guess for the fine
    # grid.  I5 governs RETURNED values: every fine point is checked below, and a fine point that
    # does not converge is re-solved and checked again.  A coarse point that did not converge is
    # therefore not fatal here -- it is COUNTED and reported in info (2026-09-29: the EXP-3 baseline
    # met 1/1503 such points; the historical code used it silently as a start, which reproduced the
    # archive because the fine Newton still converged from it).
    if legacy_unchecked:                       # frozen baseline: the historical global criterion
        ec = None
        for sc in SOLVER.continuation:
            ec, *_ = solve_e(z[ic].real + 1j * z[ic].imag * sc, tau, g, t, p, c, e0=ec)
        ec, _A, _D, c_resid, _it = solve_e(z[ic], tau, g, t, p, c, e0=ec)
    else:                                      # M35: pointwise criterion
        ec, _A, _D, c_resid, _it = _continuation_pointwise(z[ic], tau, g, t, p, c, SOLVER.continuation, SOLVER.maxit)
    coarse_unconverged = int(np.sum(~(c_resid < accept)))
    e0 = np.interp(x, x[ic], ec.real) + 1j * np.interp(x, x[ic], ec.imag)
    m = np.empty_like(e0)
    stragglers, worst = 0, 0.0
    returned_unconverged, worst_bad = 0, 0.0
    escal = dict(depth0=0, depth1=0, depth2=0)          # stragglers only (M35)
    for s in range(0, len(z), block):
        sl = slice(s, s + block)
        zb = z[sl]
        eb, left = newton_active(zb, tau, g, t, p, c, e0[sl])
        if left.size:
            stragglers += int(left.size)
            if legacy_unchecked:
                # FROZEN-BASELINE ONLY (D24): the historical fast routine re-solved stragglers with the
                # reference solver and used the result WITHOUT checking it.  Reproduced here so the
                # migration can match the archive, but every unconverged returned point is COUNTED.
                ez = None
                for sc in SOLVER.continuation:
                    ez, *_ = solve_e(zb[left].real + 1j * zb[left].imag * sc, tau, g, t, p, c, e0=ez)
                eb[left], *_ = solve_e(zb[left], tau, g, t, p, c, e0=ez)
            else:                              # M35: escalation sequence; typed failure if exhausted
                _, eb[left], _A, _D, sinfo_ = stieltjes_strict(zb[left], tau, g, t, p, c, accept)
                for k_ in ("depth0", "depth1", "depth2"):
                    escal[k_] += sinfo_["escalation"][k_]
        resid, A, D = _resid(zb, tau, g, t, p, c, eb)
        mb = (g / D).sum(1)
        if legacy_unchecked:
            bad = ~(resid < accept)
            returned_unconverged += int(bad.sum())
            worst_bad = max(worst_bad, float(resid[bad].max())) if bad.any() else worst_bad
        else:
            check_converged(resid, accept, "stieltjes_dense")
            check_branch(eb, mb, "stieltjes_dense")
        worst = max(worst, float(resid.max()))
        m[sl] = mb
    return m, dict(points=len(z), stragglers=stragglers, max_resid=worst,
                   coarse_points=len(ic), coarse_unconverged=coarse_unconverged,
                   returned_unconverged=returned_unconverged, worst_unconverged_resid=worst_bad,
                   legacy_unchecked=bool(legacy_unchecked), straggler_escalation=escal)


def jacobian_m(tau, g, t, p, c, e, A, D):
    """dm/dg_j (n_z x M), count law fixed (historical expression)."""
    tau = np.asarray(tau, float); g = np.asarray(g, float)
    Ap = -c * (p[None, :] * t[None, :] ** 2 / (1.0 + c * t[None, :] * e[:, None]) ** 2).sum(1)
    Fe = 1.0 + (g[None, :] * tau[None, :] ** 2 / D ** 2).sum(1) * Ap
    Q = (g[None, :] * tau[None, :] / D ** 2).sum(1) * Ap
    dedg = (tau[None, :] / D) / Fe[:, None]
    return 1.0 / D - Q[:, None] * dedg
