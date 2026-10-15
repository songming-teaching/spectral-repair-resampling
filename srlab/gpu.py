"""srlab.gpu -- device policy, precision guard and chunked memory for the ONE GPU-eligible kernel.

WHAT IS ELIGIBLE (measured 2026-10-02, fp64-vs-fp64, this laptop):

    Stieltjes fixed-point iteration (nz=400k x K=60)   GPU  32.5 ms   CPU(1 thread)  596 ms   18.3x
    Stieltjes fixed-point iteration (nz=400k x K=100)  GPU  51.8 ms   CPU(1 thread) 1000 ms   19.3x

WHAT IS NOT (same machine, same precision):

    dense fp64 matmul      GPU 0.218 TFLOP/s   CPU(32 thr) 0.417 TFLOP/s   -> GPU LOSES ~2x
    symmetric eigvalsh     d=2400: GPU 92 ms   CPU(1 thr) 407 ms           -> only 4.4x, and eigh is
                                                                             never a bottleneck here

So the gate in `check_gate()` is measured on the FIXED-POINT KERNEL, never on matmul: a matmul-based
gate would fail at 0.5x and kill a project whose real kernel wins 18x.

PRECISION.  The charter demands per-point residual 1e-10 and step tolerance 1e-13 (fp64 territory).
fp32 (eps 1.2e-7) cannot reach those, so every GPU array here is float64/complex128 -- there is no
fp32 path in this module.  The planned mixed-precision ladder (fp32 attempt -> fp64 polish) belongs
to the M35 escalation sequence, where EVERY level is guarded by the per-point check; it is not a
blanket substitution for fp64.

MEMORY.  8 GB, shared with the OS.  A complex128 array of nz x K is nz*K*16 bytes: at nz=400k, K=120
that is 768 MB, and a kernel needs 2-3 such arrays, so chunks are mandatory.  `plan_chunks` sizes
them from the device's measured free memory and records the plan in the manifest (D25 discipline:
whatever changes the numbers gets recorded).
"""
import os

_EPS_F32 = 1.1920928955078125e-07
_EPS_F64 = 2.220446049250313e-16


class GPUUnavailable(RuntimeError):
    """No usable CUDA device + fp64.  Callers must fall back to the CPU reference path EXPLICITLY
    (recorded, never silent) -- the charter forbids silent fallback."""


def torch_or_raise():
    try:
        import torch
    except ImportError as exc:                                   # pragma: no cover
        raise GPUUnavailable(f"torch not importable: {exc}") from exc
    if not torch.cuda.is_available():
        raise GPUUnavailable(f"torch {torch.__version__} has no CUDA device "
                             f"(build {getattr(torch.version, 'cuda', None)})")
    return torch


def device_info(torch=None):
    torch = torch or torch_or_raise()
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    return dict(name=p.name, cc=f"{p.major}.{p.minor}", sm=p.multi_processor_count,
                mem_total_mb=total // 2 ** 20, mem_free_mb=free // 2 ** 20,
                l2_mb=p.L2_cache_size // 2 ** 20, torch=torch.__version__,
                cuda=torch.version.cuda, driver=os.environ.get("NVIDIA_DRIVER_VERSION"))


def require_fp64(t, what="kernel"):
    """Precision guard: refuse fp32 where the charter needs fp64 (I9-style sentinel, not a comment).

    Accepts a torch tensor (dtype compared against torch.float64/complex128) or a numpy array.
    """
    import numpy as _np
    dt = getattr(t, "dtype", None)
    allowed = ()
    try:
        import torch as _t
        allowed = (_t.float64, _t.complex128)
    except ImportError:
        pass
    allowed = allowed + (_np.float64, _np.complex128)
    if dt not in allowed:
        raise ValueError(f"{what}: dtype {t.dtype} is not fp64; the charter needs residual 1e-10 "
                         f"(fp32 eps {_EPS_F32:.1e}).  Mixed precision goes through the M35 ladder, "
                         f"where each level is per-point checked.")


def plan_chunks(nz, K, bytes_per_element=16, mem_fraction=0.5, min_chunk=4096):
    """Chunk plan for a complex128 nz x K kernel: how many z-points fit per submission.
    A kernel needs ~3 working arrays, so budget mem_fraction of free memory / 3."""
    import torch                                                # noqa: F401  (raises if unavailable)
    free, _total = torch.cuda.mem_get_info()
    budget = int(free * mem_fraction) // 3
    per_point = max(1, K * bytes_per_element * 3)
    chunk = max(min_chunk, min(nz, budget // per_point))
    return dict(nz=nz, K=K, chunk=int(chunk), n_chunks=int(-(-nz // chunk)),
                mem_free_mb=free // 2 ** 20, bytes_per_point=per_point)


# ---------------------------------------------------------------------------------------------
# The gate (the project records, correction 2)
# ---------------------------------------------------------------------------------------------
def check_gate(threshold=3.0, nz=400_000, K=60, reps=20, verbose=True):
    """Measured go/no-go, on the RIGHT kernel.  Returns dict with the ratio and the verdict.

    Gate: fixed-point iteration fp64, GPU vs CPU(1 thread) >= `threshold`.
    DO NOT point this at matmul: dense fp64 matmul measures 0.5x on this part and would fail a
    kernel that actually wins 18x.
    """
    import time

    import numpy as np
    import torch

    # The REAL kernel shape (srlab/laws + solver): t is the count-law support, so t >= 0; p is a pmf
    # on it; tau and g are the candidate spectrum.  An earlier version of this harness used negative t
    # (arange - K/2), which makes 1 + c t e pass near zero and the fixed point ill-posed -- the two
    # paths then land in different basins and the equivalence number is meaningless.  Keep it physical.
    rng = np.random.default_rng(0)
    t_n = np.arange(K, dtype=float)
    p_n = np.exp(-t_n / 5.0); p_n /= p_n.sum()
    tau_n = np.sort(rng.random(K) * 1.5 + 0.2)
    g_n = rng.random(K); g_n /= g_n.sum()
    c = 0.75
    z_n = np.linspace(0.05, 3.0, nz) + 1j * 1e-3

    def step_np(e):
        A = (p_n[None, :] * t_n[None, :] / (1.0 + c * t_n[None, :] * e[:, None])).sum(1)
        D = tau_n[None, :] * A[:, None] - z_n[:, None]
        return (g_n[None, :] * tau_n[None, :] / D).sum(1)

    def step_torch(e, z, tau, g, t, p):
        A = (p[None, :] * t[None, :] / (1.0 + c * t[None, :] * e[:, None])).sum(1)
        D = tau[None, :] * A[:, None] - z[:, None]
        return (g[None, :] * tau[None, :] / D).sum(1)

    e_n = -(g_n @ tau_n) / z_n
    for _ in range(3):
        e_n = step_np(e_n)
    t0 = time.perf_counter(); e_n = step_np(e_n); cpu_ms = (time.perf_counter() - t0) * 1e3

    z_t = torch.tensor(z_n, dtype=torch.complex128, device="cuda")
    tau_t = torch.tensor(tau_n, dtype=torch.float64, device="cuda")
    g_t = torch.tensor(g_n, dtype=torch.float64, device="cuda")
    t_t = torch.tensor(t_n, dtype=torch.float64, device="cuda")
    p_t = torch.tensor(p_n, dtype=torch.float64, device="cuda")
    e_t = -(g_t @ tau_t) / z_t
    for _ in range(3):
        e_t = step_torch(e_t, z_t, tau_t, g_t, t_t, p_t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        e_t = step_torch(e_t, z_t, tau_t, g_t, t_t, p_t)
    torch.cuda.synchronize(); gpu_ms = (time.perf_counter() - t0) / reps * 1e3

    diff = float(np.max(np.abs(e_t.cpu().numpy() - e_n)))
    ratio = cpu_ms / gpu_ms
    out = dict(nz=nz, K=K, gpu_ms=gpu_ms, cpu_ms=cpu_ms, ratio=ratio, threshold=threshold,
               max_abs_diff=diff, passed=bool(ratio >= threshold),
               kernel="stieltjes fixed-point iteration (complex128)",
               warning="matmul would measure ~0.5x here; that is not the gate")
    if verbose:
        print(f"gate [fp64 fixed-point, nz={nz}, K={K}]: GPU {gpu_ms:.2f} ms vs CPU(1 thr) {cpu_ms:.1f} ms "
              f"-> {ratio:.1f}x (threshold {threshold}x)  {'PASS' if out['passed'] else 'FAIL'}")
        print(f"      GPU/CPU max |diff| = {diff:.3e}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--nz", type=int, default=400_000)
    ap.add_argument("--K", type=int, default=60)
    a = ap.parse_args()
    print(device_info())
    print(plan_chunks(a.nz, a.K))
    r = check_gate(a.threshold, a.nz, a.K)
    raise SystemExit(0 if r["passed"] else 1)
