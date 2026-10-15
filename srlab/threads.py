"""srlab.threads -- numerical configuration: CPU threads AND the GPU execution policy (D25 / M37 / M38).

Two separate concerns, both part of the numerical configuration and both forced (never setdefault:
an inherited environment must not change the numbers):

  1. BLAS / OpenMP thread count.  Changes LAPACK results in the last ulps and, through the flat
     Table I inversion optimum, chain-level statistics (D25).  Frozen per workload; applied before
     numpy is imported, which is why this module imports nothing numerical.

  2. GPU policy.  The GPU takes exactly ONE kind of work -- the memory-bound Stieltjes fixed-point
     iteration (fp64), which is the only source of hour-scale tail latency in this project.  Dense
     fp64 linear algebra stays on the CPU: measured 0.218 TFLOP/s on this laptop GPU vs 0.417 on the
     CPU (fp64:fp32 = 1:32 on a consumer part).  Eigh is not a bottleneck at our d (0.03-0.4 s).

`pin(workload)` must be the first thing every entry point does.

    from srlab.threads import pin; pin("tableI")        # 8 BLAS threads
    from srlab.threads import pin; pin("default")       # 1 BLAS thread, 24 CPU workers
"""
import os

THREADS = {
    "default": 1,     # everything that runs 1 process per physical core (EXP-1/2/3, S7, checks, timing)
    "tableI": 8,      # Table I: the archive was produced with 8 BLAS threads -- archive continuity (M37)
}
VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

# i9-14900HX: 24 physical cores (8P + 16E) / 32 logical.  BLAS at 32 logical usually LOSES to 24
# (hyperthread contention on FP work), so the worker count is set from PHYSICAL cores.
PHYSICAL_CORES = 24
LOGICAL_CORES = 32


def pin(workload="default"):
    """Force the BLAS thread count.  Call as the FIRST thing in every entry point.

    Three cases:
      * numpy not yet imported          -> set the variables (the only fully reliable path);
      * numpy imported, variables UNSET -> BLAS has not been told a count and has not built its thread
        pool against one; setting them now is effective for all later calls, so allow it and report
        it (debug scripts and notebooks hit this path constantly);
      * numpy imported, variables SET to a DIFFERENT count -> the pool is already built; refuse.  A
        silent switch would change the numbers (D25), so this is a hard error, not a warning.
    """
    n = str(THREADS[workload])
    if "numpy" in __import__("sys").modules:
        current = [os.environ.get(v) for v in VARS[:3]]
        if all(c == n for c in current):
            return int(n)
        if all(c is None for c in current):
            for v in VARS:
                os.environ[v] = n
            return int(n)
        raise RuntimeError(f"srlab.threads.pin('{workload}'): numpy already imported and the BLAS "
                           f"thread count is already set to {current}; switching to {n} would change "
                           f"the numbers (D25).  Pin threads before importing numpy in this process.")
    for v in VARS:
        os.environ[v] = n
    return int(n)


def workers(requested=None):
    """Worker-process count.  Default = physical cores (not logical): hyperthreads hurt FP work."""
    if requested:
        return max(1, min(PHYSICAL_CORES * 2, int(requested)))
    return max(1, PHYSICAL_CORES)


def environment_block():
    """What every manifest records (M37): the numerical configuration in force."""
    return {v: os.environ.get(v) for v in VARS}
