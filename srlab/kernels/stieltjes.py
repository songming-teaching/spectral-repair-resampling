"""srlab.kernels.stieltjes -- the GPU fixed-point kernel (M38, stages 2a-2c).

THE ONLY GPU-ELIGIBLE KERNEL.  Everything else stays on the CPU (see srlab/gpu.py for why: dense fp64
matmul measures 0.5x on this part, eigh 4.4x and is never a bottleneck).

The iteration is the same Newton update as `srlab.solver.solve_e_pointwise`, in complex128:

    den = 1 + c t e          (nz x K)
    A   = sum_k p_k t_k / den
    D   = tau_j A - z        (nz x M)
    F   = sum_j g_j tau_j / D
    Ap  = -c sum_k p_k t_k^2 / den^2
    dF  = -(sum_j g_j tau_j^2 / D^2) Ap
    e  <- e - (e - F) / (1 - dF)                 (Newton step)
    e  <- 0.5 e + 0.5 F  where Im(e) <= 0 or non-finite   (branch repair, as history)

Two properties are non-negotiable, both inherited:

  1. PER-POINT STOPPING and ACTIVE-SET SHRINKAGE (M35).  The historical global criterion stopped when
     the MAXIMUM step was small, so every point iterated as long as the slowest one (the audit's cost
     source) and could stop with an individual point unconverged (the silent source, D24).  Here each
     point leaves the active set on its own criterion, and the active set is a GPU index tensor, so
     per-iteration work shrinks as points finish.  Measured on the hardest project case: the active
     set goes 20000 -> 2 -> 0 over 21 iterations.

  2. CHUNKING, and what it does NOT preserve.  8 GB of device memory shared with the OS.  Chunks bound
     memory -- but they DO move the last ulps, measured 2026-10-02:

         K=21 : chunk 20000 vs 4096 vs 777  -> bitwise identical
         K=114: same sizes                  -> differ in the last ulps (5e-17 relative), and the
                                               iteration count moves (CPU 233, GPU 306)

     Reason: torch picks a reduction kernel by tensor shape, the shapes change as the active set
     shrinks, and float summation is not associative.  The pointwise test compares |en - ea| against
     tol*(1+|ea|) with tol = 1e-13, i.e. at the noise floor of the CPU-vs-GPU arithmetic difference,
     so a point may be judged converged one iteration earlier or later.  The FIXED POINT is unaffected
     (both paths agree to ~1e-15 relative).

     Consequence, now config policy: chunk_z is FROZEN in config.GPUCfg, never sized from momentary
     free memory -- otherwise two runs of the same experiment could differ.  Free-memory sizing
     survives only as an explicit fallback, recorded in the manifest.

MEMORY NOTE.  `D` is (nz, M) complex; at nz=400k, M=400 that is 2.6 GB, so it is NOT materialised
unless `want_AD=True`.  Callers on the dense path want `m` only.  (A first version returned D
unconditionally -- a latent out-of-memory bug at production sizes, fixed here.)
"""
import numpy as np

from ..gpu import plan_chunks, require_fp64


def _to_torch(z, tau, g, t, p, c):
    import torch
    dev = "cuda"
    z_t = torch.as_tensor(np.asarray(z, dtype=np.complex128), device=dev)
    tau_t = torch.as_tensor(np.asarray(tau, dtype=np.float64), device=dev)
    g_t = torch.as_tensor(np.asarray(g, dtype=np.float64), device=dev)
    t_t = torch.as_tensor(np.asarray(t, dtype=np.float64), device=dev)
    p_t = torch.as_tensor(np.asarray(p, dtype=np.float64), device=dev)
    for name, arr in (("z", z_t), ("tau", tau_t), ("g", g_t), ("t", t_t), ("p", p_t)):
        require_fp64(arr, f"stieltjes kernel argument {name}")
    if z_t.dtype != torch.complex128:
        raise ValueError(f"z must be complex128, got {z_t.dtype}")
    return z_t, tau_t, g_t, t_t, p_t, float(c)


def solve_e_gpu(z, tau, g, t, p, c, e0=None, maxit=None, tol=None, chunk=None, budget=None,
                want_AD=False, record_traj=True):
    """Pointwise-criterion fixed point on the GPU, chunked over z.

    Returns (m, e, A, D, resid, info).  A and D are None unless want_AD (see MEMORY NOTE above).

    `budget` (srlab.tail.Budget, optional) is checked once per iteration per chunk; on exhaustion it
    raises TailBudgetExceeded, a typed event the experiment layer maps to a registered abstention.
    `info['unconverged_at_maxit']` counts points that hit the cap, so a caller can never mistake an
    unconverged return for a converged one.
    """
    import torch

    from ..config import GPU, SOLVER
    maxit = SOLVER.maxit if maxit is None else maxit
    tol = SOLVER.tol if tol is None else tol

    z = np.asarray(z, complex)
    tau_a = np.asarray(tau, float); g_a = np.asarray(g, float)
    t_a = np.asarray(t, float); p_a = np.asarray(p, float)

    # zero-weight atoms contribute to neither sum: dropping them is exact (same as the CPU path)
    keep = g_a > 0
    tau_a, g_a = tau_a[keep], g_a[keep]
    M = len(tau_a)

    if chunk is None:
        chunk = GPU.chunk_z
    if chunk is None:
        plan = plan_chunks(len(z), len(t_a))
        plan["sized_from"] = "free memory (fallback: chunk not pinned in config)"
    else:
        plan = dict(nz=len(z), K=len(t_a), chunk=int(chunk), n_chunks=int(-(-len(z) // int(chunk))),
                    mem_free_mb=None, bytes_per_point=None, sized_from="config.GPUCfg.chunk_z (frozen)")
    step = plan["chunk"]

    e_out = np.empty(len(z), complex)
    m_out = np.empty(len(z), complex)
    resid_out = np.empty(len(z))
    A_out = np.empty((len(z), len(t_a)), complex) if want_AD else None
    D_out = np.empty((len(z), M), complex) if want_AD else None
    per_chunk_iters, act_traj, tot_unc = [], [], 0

    for s in range(0, len(z), step):
        sl = slice(s, s + step)
        z_t, tau_t, g_t, t_t, p_t, cc = _to_torch(z[sl], tau_a, g_a, t_a, p_a, c)
        if e0 is None:
            a1 = float(g_a @ tau_a)
            e = (-a1 / z_t)
        else:
            e = torch.as_tensor(np.asarray(e0[sl], dtype=np.complex128), device="cuda")

        act = torch.arange(z_t.numel(), device="cuda")
        traj = []
        it = 0
        for it in range(1, maxit + 1):
            if budget is not None:
                budget.check()
            full = act.numel() == e.numel()
            ea = e if full else e[act]
            za = z_t if full else z_t[act]
            den = 1.0 + cc * t_t[None, :] * ea[:, None]
            A = (p_t[None, :] * t_t[None, :] / den).sum(1)
            D = tau_t[None, :] * A[:, None] - za[:, None]
            F = (g_t[None, :] * tau_t[None, :] / D).sum(1)
            Ap = -cc * (p_t[None, :] * t_t[None, :] ** 2 / den ** 2).sum(1)
            dF = -(g_t[None, :] * tau_t[None, :] ** 2 / D ** 2).sum(1) * Ap
            en = ea - (ea - F) / (1.0 - dF)
            bad = (en.imag <= 0) | ~torch.isfinite(en)
            if bool(bad.any()):
                en = torch.where(bad, 0.5 * ea + 0.5 * F, en)
            if full:
                e = en
            else:
                e[act] = en                    # in place: `ea`, `F` were already extracted above
            done = (en - ea).abs() < tol * (1.0 + ea.abs())
            act = torch.nonzero(~done, as_tuple=False).flatten() if full else act[~done]
            if record_traj:
                traj.append(int(act.numel()))
            if act.numel() == 0:
                break
        per_chunk_iters.append(it)
        act_traj.append(traj)
        tot_unc += int(act.numel())
        if budget is not None:
            budget.note(iterations=it, points=int(z_t.numel()), left_unconverged=int(act.numel()))

        Ae = (p_t[None, :] * t_t[None, :] / (1.0 + cc * t_t[None, :] * e[:, None])).sum(1)
        De = tau_t[None, :] * Ae[:, None] - z_t[:, None]
        resid = (e - (g_t[None, :] * tau_t[None, :] / De).sum(1)).abs()
        e_out[sl] = e.cpu().numpy()
        m_out[sl] = (g_t[None, :] / De).sum(1).cpu().numpy()
        resid_out[sl] = resid.cpu().numpy()
        if want_AD:
            A_out[sl] = Ae.cpu().numpy()
            D_out[sl] = De.cpu().numpy()
        del z_t, tau_t, g_t, t_t, p_t, e, Ae, De, resid
        torch.cuda.empty_cache()

    info = dict(points=len(z), K=len(t_a), M=M, chunk=step, n_chunks=plan["n_chunks"], plan=plan,
                per_chunk_iters=per_chunk_iters, active_set=act_traj if record_traj else None,
                unconverged_at_maxit=tot_unc, max_resid=float(np.max(resid_out)),
                backend="gpu", dtype="complex128", solver_tol=tol, maxit=maxit)
    return m_out, e_out, A_out, D_out, resid_out, info
