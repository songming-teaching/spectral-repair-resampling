"""srlab.forward -- the forward map: quantiles of mu = G boxtimes T_c(nu)  (M4 + the M17 fix).

The density of mu is read off the Stieltjes transform at an ABSOLUTE smoothing scale
Im z = eta = eta_rel * a1*m1(nu) (M17 iteration 3: absolute, spectrum-scaled, window-independent),
the window grows until the numerical first moment captures a1*m1(nu) (I3), and the grid spacing is
required to resolve the smoothing scale, dx <= eta (I6).

Two solver paths share ONE implementation of the window / grid / quadrature rule, exactly as in the
historical code (M18):
  * `law_quantiles_ref` -- the reference solver (`solver.stieltjes`), kept as the test oracle;
  * `law_quantiles`     -- warm start + active-set Newton on the dense grid (`solver.stieltjes_dense`).

I6 / cap policy (M26, registered 2026-09-28).  The historical 4e5 cap is genuinely hit in the
S7 regime (H = delta_1, heavy tails): n_needed/n_cap = 1.16 / 1.39 / 2.68 at c = 0.5 / 0.75 / 1.25,
i.e. dx > eta by up to 2.7x -- yet the M17 precision criterion (m1 capture) still passes, so no
number is wrong; the GUARD is what is being violated in silence.  Under M26 the default is
cap_policy="report": the historical arithmetic is unchanged (M17/M18 results stay bitwise
reproducible) but the shortfall is recorded.  Anything that records a violation is a MANDATORY
manifest field here -- `info["cap"]` is part of the return contract, not an optional diagnostic,
and srlab.run refuses to write a manifest that omits it.
"""
from contextlib import contextmanager

import numpy as np

from .config import QUANT
from . import solver
from .invariants import InvariantError, check_converged, require


def law_zero_atom(g, tau, p_nu0, c):
    """Zero atom of the limit law: the larger of the fitted atom at 0 and the count-rank identity."""
    g0 = float(g[0]) if abs(float(tau[0])) < 1e-12 else 0.0
    return max(g0, max(0.0, 1.0 - (1.0 - p_nu0) / c))


def _solve_dense_path(zz, tau, g, t, p, c, legacy_unchecked):
    """O-3: choose the dense solver and REPORT the choice.

    The GPU kernel (kernels.dense) is the default for the production path.  It is NOT used when
    `legacy_unchecked` is set -- that is the frozen pre-M19 baseline, which needs the historical
    global criterion and an unchecked re-solve the GPU ladder deliberately does not implement, so
    routing it to the GPU would silently change the baseline.  It is also not used below
    `GPU.min_points_for_gpu` points, where launch overhead and the shape-dependent reduction make the
    CPU path preferable, or when no CUDA device is present.

    Returns (m, sinfo, path).  `path` is one of "gpu" / "cpu-fallback:<reason>" and MUST reach the
    manifest: a number produced on the fallback path is still admissible, but it must be recorded as
    such (D25), otherwise the ledger cannot tell which backend produced it.  Falling back is never
    silent -- the reason string is carried, not a flag that loses why.
    """
    from .config import GPU
    if legacy_unchecked:
        return (*solver.stieltjes_dense(zz, tau, g, t, p, c, QUANT.n_coarse, QUANT.block,
                                        legacy_unchecked=True), "cpu-fallback:legacy_baseline")
    if not GPU.enabled:
        return (*solver.stieltjes_dense(zz, tau, g, t, p, c, QUANT.n_coarse, QUANT.block,
                                        legacy_unchecked=False), "cpu-fallback:gpu_disabled")
    if len(zz) < GPU.min_points_for_gpu:
        return (*solver.stieltjes_dense(zz, tau, g, t, p, c, QUANT.n_coarse, QUANT.block,
                                        legacy_unchecked=False),
                f"cpu-fallback:small_grid({len(zz)}<{GPU.min_points_for_gpu})")
    try:
        from .gpu import torch_or_raise
        torch_or_raise()
    except Exception as exc:                                                  # noqa: BLE001
        return (*solver.stieltjes_dense(zz, tau, g, t, p, c, QUANT.n_coarse, QUANT.block,
                                        legacy_unchecked=False),
                f"cpu-fallback:no_cuda({type(exc).__name__})")
    from .kernels import dense as kd
    from .tail import Budget
    attempts = []
    with Budget(GPU.tail_seconds, "law_quantiles", attempts) as b:
        m, sinfo = kd.stieltjes_dense_gpu(zz, tau, g, t, p, c, budget=b)
    return m, sinfo, "gpu"


@contextmanager
def pin_cpu_backend():
    """Force the dense solve onto the CPU for the duration of the block.

    WHY THIS IS A PUBLIC TOOL AND NOT THREE LOCAL HACKS.  Since the GPU path became the default,
    "does the migrated code reproduce the archive?" and "is the GPU path equivalent to the CPU one?"
    are two DIFFERENT questions with two different answers:

      * the archive (`archive/code_2026-09-27`, `data/m17/verify_quantile_fix.json`) was produced by
        numpy on the CPU, so only a CPU-vs-CPU comparison can be bitwise -- that is the migration
        claim, and it stays at 1e-12;
      * the GPU path differs in floating-point reduction order, which shows up at ~1e-11 on the
        hardest cases (the 4e5-cap ones), five orders inside the M18 1e-5 tolerance but far outside a
        bitwise assertion.

    Without a named way to say which question is being asked, the natural fix -- loosening the
    bitwise assertion until it passes -- would quietly retire the migration guarantee.  Pinning is
    explicit, local, and restores the previous backend rather than assuming the caller's state."""
    from . import solver as _solver
    from .config import QUANT as _Q
    orig = globals()["_solve_dense_path"]

    def cpu_only(zz, tau_, g_, t_, p_, c_, legacy_unchecked):
        m, sinfo = _solver.stieltjes_dense(zz, tau_, g_, t_, p_, c_, _Q.n_coarse, _Q.block,
                                           legacy_unchecked=legacy_unchecked)
        return m, dict(sinfo, path="cpu-pinned"), "cpu-pinned"
    globals()["_solve_dense_path"] = cpu_only
    try:
        yield
    finally:
        globals()["_solve_dense_path"] = orig


def _quantile_routine(solve_dense, tau, g, t, p, c, u, n_x, eta_rel, x_hi, m1_tol, cap_policy, legacy_unchecked=False):
    tau = np.asarray(tau, float); g = np.asarray(g, float)
    t = np.asarray(t, float); p = np.asarray(p, float)
    p0 = law_zero_atom(g, tau, float(p[0]), c)
    mean = float(g @ tau) * float(p @ t)          # int x dmu  (first-moment multiplicativity)
    if x_hi is None:
        x_hi0 = max(1.05 * float(np.max(tau[g > 1e-12])) * (1.0 + np.sqrt(c)) ** 2, 1e-3)
        grow = True
    else:
        x_hi0 = float(x_hi)
        grow = False
    cap = QUANT.n_x_cap

    def _one(eta_rel_l):
        """One full window/grid/quadrature pass at a GIVEN eta_rel.  Returns (q, info, diagnostics)."""
        import time as _time
        t_start = _time.time()
        eta = eta_rel_l * mean
        x_hi = x_hi0
        worst_ratio = 0.0; cap_exceeded = 0; acc_unconv = 0; paths_seen = set()
        sinfo = None
        while True:
            n_needed = int(x_hi / eta) + 1
            n_used = min(max(n_x, n_needed), cap)
            ratio = n_needed / n_used
            worst_ratio = max(worst_ratio, ratio)
            if ratio > 1.0:
                cap_exceeded += 1
                if cap_policy == "raise":
                    require(False, "I6",
                            f"law_quantiles: grid spacing dx > eta (need {n_needed} points, cap {cap}; "
                            f"ratio {ratio:.3f}); raise the cap or switch to an adaptive grid")
            xs = np.linspace(1e-6 * mean, x_hi, n_used)
            zz = xs + 1j * eta
            if solve_dense:
                m, sinfo, path = _solve_dense_path(zz, tau, g, t, p, c, legacy_unchecked)
                acc_unconv = acc_unconv + sinfo["returned_unconverged"]
                paths_seen.add(path)
            else:
                m = solver.stieltjes(zz, tau, g, t, p, c)[0]
                sinfo = None
            f = np.maximum((m + p0 / zz).imag, 0.0) / np.pi
            mass = float(np.trapezoid(f, xs))
            mom = float(np.trapezoid(xs * f, xs))
            if not grow or mom > (1 - m1_tol) * mean or x_hi > QUANT.x_hi_max_rel * mean:
                break
            x_hi *= 1.5
        F = p0 + np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xs))])
        tot = F[-1]
        capture_mass = (tot - p0) / max(1.0 - p0, 1e-300)
        Fo = p0 + (F - p0) * (1.0 - p0) / max(tot - p0, 1e-12)   # I9: gap = 1 - capture_mass
        q = np.interp(u, Fo, xs, left=0.0, right=xs[-1])
        q[u <= p0] = 0.0
        info = dict(p0=p0, eta=eta, eta_rel=eta_rel_l, mean=mean, x_hi=x_hi, n_used=len(xs),
                    m1_capture=mom / mean, density_mass=mass, mass_capture=capture_mass,
                    window_grown=bool(grow), m1_ok=bool(abs(mom / mean - 1.0) <= m1_tol),
                    sec=round(_time.time() - t_start, 3),
                    cap=dict(policy=cap_policy, n_cap=cap, worst_dx_over_eta=worst_ratio,
                             iterations_over_cap=cap_exceeded, exceeded=bool(cap_exceeded)))
        if sinfo is not None:
            info["solver"] = dict(sinfo, returned_unconverged_all_iterations=acc_unconv)
            info["solver"]["path"] = (sorted(paths_seen)[0] if len(paths_seen) == 1
                                      else "mixed:" + "+".join(sorted(paths_seen)))
            info["solver"]["paths_seen"] = sorted(paths_seen)
        return q, info, capture_mass, mass, mom

    # --- eta-refinement ladder (recorded design decision, item 1) --------------------------------------------
    # eta is a NUMERICAL refinement scale (Im z = eta): eta -> 0 converges to the SAME limit object,
    # NOT a model-parameter change, so the paper needs no eta_rel declaration -- S4 carries one line:
    # "when the mass guard fires, eta is refined along the ladder".  The GLOBAL default eta_rel is
    # untouched (no M17-grade re-acceptance); only an I3-failing point refines LOCALLY, one rung at a
    # time (÷5), until the guard passes or the ladder is exhausted.  This is the M26 philosophy
    # ("local escalation, not a global parameter change") applied to the other knob.
    #
    # `legacy_unchecked` is the ARCHIVE-reproduction path: the archive had no I3 check at all, so it
    # must take the FIRST rung's value with no refinement (recorded design decision, item 2 -- these cells are a
    # known archived defect, exempt from bitwise regression with a documented waiver).
    if legacy_unchecked:
        levels = [eta_rel]
    else:
        levels = [eta_rel, eta_rel / 5.0, eta_rel / 25.0]
    ladder = []
    last = None
    for i, erl in enumerate(levels):
        q, info, capture_mass, mass, mom = _one(erl)
        ladder.append(dict(level=i, eta_rel=erl, n_used=info["n_used"], mass_capture=capture_mass,
                           m1_capture=info["m1_capture"], sec=info["sec"]))
        last = (q, info, capture_mass, mass, mom)
        if legacy_unchecked:
            info["i3_skipped"] = bool(grow)          # archive defect kept, but RECORDED (not hidden)
            info["eta_ladder"] = ladder
            break
        if not grow:                                 # window fixed by the caller: guard not enforced
            info["i3_skipped"] = False
            info["eta_ladder"] = ladder
            break
        try:
            require(abs(capture_mass - 1.0) <= m1_tol, "I3",
                    f"mass capture {capture_mass:.6f} differs from 1 by more than m1_tol = {m1_tol:.1e}")
            require(mass <= 1.0 - p0 + 1e-9 / max(1.0 - p0, 1e-300), "I3",
                    f"density mass {mass:.6f} exceeds the available 1 - p0 = {1.0 - p0:.6f}")
            check_converged(np.array([abs(mom / mean - 1.0)]), m1_tol, "I3 first-moment capture")
        except Exception as exc:                                            # noqa: BLE001
            if i < len(levels) - 1:
                continue                            # refine eta and retry (local escalation)
            info["eta_ladder"] = ladder
            raise InvariantError(f"[I3] eta ladder exhausted ({len(levels)} rungs, eta_rel down to "
                                 f"{levels[-1]:.1e}): {exc}") from exc
        info["i3_skipped"] = False
        info["eta_ladder"] = ladder
        break
    q, info, capture_mass, mass, mom = last
    return q, info


def law_quantiles_ref(tau, g, t, p, c, u, n_x=QUANT.n_x_min, eta_rel=QUANT.eta_rel,
                      x_hi=None, m1_tol=QUANT.m1_tol, cap_policy=QUANT.cap_policy):
    """Quantile function Q(u) of mu = G boxtimes T_c(nu) -- reference solver (M17 version, M18 oracle)."""
    return _quantile_routine(False, tau, g, t, p, c, u, n_x, eta_rel, x_hi, m1_tol, cap_policy)


def law_quantiles(tau, g, t, p, c, u, n_x=QUANT.n_x_min, eta_rel=QUANT.eta_rel,
                  x_hi=None, m1_tol=QUANT.m1_tol, cap_policy=QUANT.cap_policy, legacy_unchecked=False):
    """Same quantity, window, grid and quadrature as `law_quantiles_ref`; only the fixed-point solve
    on the dense grid is accelerated (M18: warm start + active-set Newton, equal to the reference
    to 1e-15)."""
    return _quantile_routine(True, tau, g, t, p, c, u, n_x, eta_rel, x_hi, m1_tol, cap_policy, legacy_unchecked)


def w1_from_quantiles(qa, qb, u):
    """W1 between two laws given their quantile functions on the same u-grid."""
    return float(np.trapezoid(np.abs(qa - qb), u))
