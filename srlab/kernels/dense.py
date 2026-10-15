"""srlab.kernels.dense -- the dense-grid GPU path: what `law_quantiles` actually calls (stage 2d).

`srlab.solver.stieltjes_dense` has three stages; this reproduces all three on the GPU:

    (1) coarse points by continuation in Im z        -> initial guess
    (2) linear interpolation onto the fine grid
    (3) per-chunk active-set Newton on the fine grid; failures climb the PRECISION LADDER, and every
        rung is per-point checked in fp64 before its result is accepted

The M38 precision ladder (stage 2b), one rung at a time, each handling only the points still failing:

    L1  complex64 sweep          cheap; accepted only if the fp64 residual check passes
    L2  complex128 pointwise     the validated stage-2a kernel
    L3  denser continuation      the M35 depth-1 schedule
    L4  10x iteration cap        the M35 depth-2 schedule

A low-precision rung can never leak into a returned value: the check is fp64 and path-independent
(I5 is checked on the RESULT, not claimed from the path).  Where fp32 cannot reach the tolerance -- near
the spectral edge the contraction factor approaches 1 -- the rung fails its check and fp64 does the work.
Stage 2b measures exactly which cases those are.

Every loop calls `budget.check()`; the ladder exhausting raises SolverEscalationFailure, the budget
expiring raises TailBudgetExceeded.  Both are ConvergenceError, so the experiment layer turns them into
registered abstentions (M21) instead of aborting a campaign.
"""
import numpy as np

from ..invariants import SolverEscalationFailure, check_branch, check_converged
from ..solver import ESCALATION
from . import stieltjes as K


def residual_fp64(z, tau, g, t, p, c, e):
    """The charter measure, computed in fp64 on the CPU from the returned e -- path-independent."""
    den = 1.0 + c * np.asarray(t)[None, :] * np.asarray(e)[:, None]
    A = (np.asarray(p)[None, :] * np.asarray(t)[None, :] / den).sum(1)
    D = np.asarray(tau)[None, :] * A[:, None] - np.asarray(z)[:, None]
    return np.abs(np.asarray(e) - (np.asarray(g)[None, :] * np.asarray(tau)[None, :] / D).sum(1))


def _device(dtype):
    import torch
    r = torch.float32 if dtype == "fp32" else torch.float64
    c = torch.complex64 if dtype == "fp32" else torch.complex128
    return r, c


def _iterate(z_pt, tau_a, g_a, t_a, p_a, c, e, maxit, tol, dtype, budget):
    """Pointwise Newton on the device.  Returns (e_np_complex128, iterations, traj, left_unconverged)."""
    import torch
    rd, cd = _device(dtype)
    z_t = torch.as_tensor(np.asarray(z_pt, complex), dtype=cd, device="cuda")
    tau_t = torch.as_tensor(tau_a, dtype=rd, device="cuda")
    g_t = torch.as_tensor(g_a, dtype=rd, device="cuda")
    t_t = torch.as_tensor(t_a, dtype=rd, device="cuda")
    p_t = torch.as_tensor(p_a, dtype=rd, device="cuda")
    e_t = torch.as_tensor(np.asarray(e, complex), dtype=cd, device="cuda")
    act = torch.arange(z_t.numel(), device="cuda")
    traj = []
    it = 0
    for it in range(1, maxit + 1):
        if budget is not None:
            budget.check()
        full = act.numel() == e_t.numel()
        ea = e_t if full else e_t[act]
        za = z_t if full else z_t[act]
        den = 1.0 + c * t_t[None, :] * ea[:, None]
        A = (p_t[None, :] * t_t[None, :] / den).sum(1)
        D = tau_t[None, :] * A[:, None] - za[:, None]
        F = (g_t[None, :] * tau_t[None, :] / D).sum(1)
        Ap = -c * (p_t[None, :] * t_t[None, :] ** 2 / den ** 2).sum(1)
        dF = -(g_t[None, :] * tau_t[None, :] ** 2 / D ** 2).sum(1) * Ap
        en = ea - (ea - F) / (1.0 - dF)
        bad = (en.imag <= 0) | ~torch.isfinite(en)
        if bool(bad.any()):
            en = torch.where(bad, 0.5 * ea + 0.5 * F, en)
        if full:
            e_t = en
        else:
            e_t[act] = en
        done = (en - ea).abs() < tol * (1.0 + ea.abs())
        act = torch.nonzero(~done, as_tuple=False).flatten() if full else act[~done]
        traj.append(int(act.numel()))
        if act.numel() == 0:
            break
    out = e_t.to(torch.complex128).cpu().numpy()
    left = int(act.numel())
    del z_t, tau_t, g_t, t_t, p_t, e_t, act
    torch.cuda.empty_cache()
    return out, it, traj, left


def _solve_chunked(z_pt, tau_a, g_a, t_a, p_a, c, e_start, maxit, tol, dtype, chunk, budget):
    """Apply `_iterate` over z-chunks with a warm start per chunk."""
    outs, its, lefts, trajs = [], [], [], []
    for s in range(0, len(z_pt), chunk):
        sl = slice(s, s + chunk)
        o, it, traj, left = _iterate(z_pt[sl], tau_a, g_a, t_a, p_a, c, e_start[sl], maxit, tol, dtype, budget)
        outs.append(o); its.append(it); lefts.append(left); trajs.append(traj)
    return np.concatenate(outs), max(its), sum(lefts), trajs


def stieltjes_dense_gpu(z, tau, g, t, p, c, n_coarse=None, budget=None, accept=None, chunk=None,
                        maxit=None, tol=None, ladder=True, want_AD=False):
    """GPU dense-grid Stieltjes transform.  Returns (m, info) -- same contract as the CPU path,
    plus the precision-ladder record and the tail-budget record in `info`.

    Raises TailBudgetExceeded / SolverEscalationFailure (both typed, M21)."""
    from ..config import GPU, QUANT, SOLVER
    n_coarse = QUANT.n_coarse if n_coarse is None else n_coarse
    accept = SOLVER.accept_resid if accept is None else accept
    maxit = SOLVER.maxit if maxit is None else maxit
    tol = SOLVER.tol if tol is None else tol
    chunk = GPU.chunk_z if chunk is None else chunk

    z = np.asarray(z, complex)
    x = z.real
    assert np.all(np.diff(x) > 0) and np.ptp(z.imag) == 0, "dense grid must be sorted with constant Im z"
    tau_a = np.asarray(tau, float); g_a = np.asarray(g, float)
    t_a = np.asarray(t, float); p_a = np.asarray(p, float)
    keep = g_a > 0
    tau_a, g_a = tau_a[keep], g_a[keep]

    rungs = []

    # ---------------- (1) coarse continuation + (2) interpolation --------------------------------
    stride = max(1, len(z) // n_coarse)
    ic = np.unique(np.r_[np.arange(0, len(z), stride), len(z) - 1])
    zc = z[ic]
    if budget is not None:
        budget.start_level(0, "coarse continuation")
    e_guess = np.zeros(len(ic), complex)
    coarse_iters = 0
    for sc in SOLVER.continuation:
        e_guess, it, _, _ = _iterate(zc.real + 1j * zc.imag * sc, tau_a, g_a, t_a, p_a, c, e_guess,
                                     maxit, tol, "fp64", budget)
        coarse_iters = max(coarse_iters, it)
    e_guess, it, _, _ = _iterate(zc, tau_a, g_a, t_a, p_a, c, e_guess, maxit, tol, "fp64", budget)
    coarse_iters = max(coarse_iters, it)
    r_c = residual_fp64(zc, tau_a, g_a, t_a, p_a, c, e_guess)
    coarse_unc = int(np.sum(~(r_c < accept)))
    if budget is not None:
        budget.note(iterations=coarse_iters, points=len(ic), left_unconverged=coarse_unc,
                    max_resid=float(np.max(r_c)))
        budget.finish("converged" if coarse_unc == 0 else "partial")
    rungs.append(dict(level=0, label="coarse continuation", points=len(ic), left_unconverged=coarse_unc,
                      max_resid=float(np.max(r_c)), iterations=coarse_iters))
    e = np.interp(x, zc.real, e_guess.real) + 1j * np.interp(x, zc.real, e_guess.imag)

    # ---------------- (3) fine grid: the precision ladder ----------------------------------------
    # Stage 2b measurement: the fp32 rung CANNOT satisfy the 1e-13 pointwise criterion (fp32 eps is
    # 1.2e-7), so every point survives it and fp64 finishes in ~2 iterations from the coarse warm start.
    # It rescued 0/60000 points.  a recorded design decision O-2(a): the rung is DISABLED by default
    # (config.GPUCfg.fp32_sweep_maxit = 0).  The rung is dropped from the plan entirely rather than run
    # with maxit=0 -- an empty sweep would still ship the whole array to the device, compute a residual
    # and leave a zero-iteration "L1" record in the manifest, i.e. it would LOOK like a rung that ran
    # and lost.  A rung that cannot earn its place must be absent from the record, not present and empty.
    fp32_on = GPU.fp32_sweep_maxit > 0
    plan = ([("fp32", "L1 fp32 sweep", None)] if fp32_on else []) + [("fp64", "L2 fp64 pointwise", None)]
    if ladder:
        plan += [("fp64", "L3 denser continuation", ESCALATION[0]),
                 ("fp64", "L4 10x cap", ESCALATION[1])]
    left = len(z)
    for lvl, (dtype, label, spec) in enumerate(plan, start=1):
        if budget is not None:
            budget.start_level(lvl, label)
        if dtype == "fp32":
            mi_override = GPU.fp32_sweep_maxit
        else:
            mi_override = None
        if spec is not None:                       # densify the continuation: same warm start chain
            e_g = e
            for sc in spec["levels"]:
                e_g, _, _, _ = _solve_chunked(z.real + 1j * z.imag * sc, tau_a, g_a, t_a, p_a, c, e_g,
                                              spec["maxit"], tol, "fp64", chunk, budget)
            e = e_g
            mi = maxit if spec is None else spec["maxit"]
        else:
            mi = maxit
        e, it, left, _ = _solve_chunked(z, tau_a, g_a, t_a, p_a, c, e,
                                        mi if mi_override is None else mi_override,
                                        tol, dtype, chunk, budget)
        r = residual_fp64(z, tau_a, g_a, t_a, p_a, c, e)
        left = int(np.sum(~(r < accept)))
        if budget is not None:
            budget.note(iterations=it, points=len(z), left_unconverged=left,
                        max_resid=float(np.max(r)), path="gpu+" + dtype)
        rungs.append(dict(level=lvl, label=label, dtype=dtype, points=len(z), left_unconverged=left,
                          max_resid=float(np.max(r)), iterations=it))
        if left == 0:
            if budget is not None:
                budget.finish("converged")
            break
    else:
        if budget is not None:
            budget.finish("stalled")
        raise SolverEscalationFailure(
            f"[I5/M38] dense GPU ladder exhausted: {left}/{len(z)} points unconverged "
            f"(worst resid {float(np.max(r)):.3e})",
            dict(ladder=rungs, tail=budget.manifest_block() if budget is not None else None))

    den = 1.0 + c * t_a[None, :] * e[:, None]
    A = (p_a[None, :] * t_a[None, :] / den).sum(1)
    D = tau_a[None, :] * A[:, None] - z[:, None]
    resid = np.abs(e - (g_a[None, :] * tau_a[None, :] / D).sum(1))
    check_converged(resid, accept, "stieltjes_dense_gpu")
    m = (g_a[None, :] / D).sum(1)
    check_branch(e, m, "stieltjes_dense_gpu")

    # --- the CPU-path diagnostic contract (stieltjes_dense), so callers need no branch on backend ---
    # forward.py reads info["returned_unconverged"] directly and supervisor reads the rest; a GPU info
    # dict missing those keys would either KeyError or silently drop the diagnostics from the ledger.
    # We got here only by converging every point (check_converged above), so the unchecked-return
    # counters are zero BY CONSTRUCTION, not by assumption -- and they are stated rather than omitted.
    l2 = next((r for r in rungs if r["label"].startswith("L2 ")), None)
    info = dict(points=len(z), K=len(t_a), M=len(tau_a), chunk=chunk,
                n_chunks=int(-(-len(z) // chunk)), coarse_points=len(ic), coarse_unconverged=coarse_unc,
                max_resid=float(np.max(resid)), unconverged_at_maxit=0, ladder=rungs, backend="gpu",
                tail=budget.manifest_block() if budget is not None else None,
                A=A if want_AD else None, D=D if want_AD else None)
    info.update(returned_unconverged=0, worst_unconverged_resid=0.0, legacy_unchecked=False,
                stragglers=int(l2["left_unconverged"]) if l2 is not None else 0,
                straggler_escalation=dict(depth0=0, depth1=0, depth2=0))
    return m, info
