"""Inversion tests (P1 gap found 2026-09-28: invert() had never been exercised end-to-end, and
Problem.model() read two SolverCfg fields from InversionCfg -- it failed on the SECOND objective
evaluation).  These tests run complete inversions.

  I1  a full invert() on an EXP-1-like case recovers the two-point H0 to the historical accuracy;
  I2  the migrated invert() equals the archive srinv.invert on identical arguments (legacy baseline);
  I3  invert_or_abstain maps a forced solver failure to a registered abstention (M21), never an abort.

Run:  python tests/test_inverse.py
"""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from srlab import inverse, laws, sampling                           # noqa: E402
from srlab.config import INV_SCALEFREE, INV_V1_BASELINE, GRID_005   # noqa: E402

ARCH = os.path.join(ROOT, "reference")


def _case(seed=20260925, N=600, alpha=1.0):
    rng = np.random.default_rng(seed)
    d = int(round(0.75 * N)); d0 = int(round(0.7 * d))
    sig = np.where(np.arange(d) < d0, 0.5, 2.5)
    lam = sampling.gamma_weights(rng, alpha, N)
    X = np.sqrt(sig)[:, None] * rng.standard_normal((d, N))
    n = sampling.multinomial(rng, lam)
    ev = sampling.spectrum(X, n)
    return ev, d, N, lam, (np.array([0.5, 2.5]), np.array([d0 / d, 1 - d0 / d]))


def test_I1_full_inversion():
    ev, d, N, lam, HN = _case()
    pr = inverse.Problem(ev, d, N, laws.plugin_multi(lam), GRID_005, cfg=INV_SCALEFREE)
    g, info = inverse.invert(pr)
    w1 = inverse.w_to(g, GRID_005, *HN, 1)
    assert info["status"] in ("success", "retry_success"), info
    assert w1 < 0.2, w1                     # EXP-1 at N=500-1000: W1 ~ 0.07-0.12
    return round(w1, 4), info["status"], info["nit"]


def _load_archive():
    sys.path.insert(0, ARCH)
    spec = importlib.util.spec_from_file_location("_arch_srinv", os.path.join(ARCH, "srinv.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_I2_equals_archive():
    ev, d, N, lam, HN = _case(N=400)
    nu = laws.plugin_multi(lam)
    A = _load_archive()
    pa = A.Problem(ev, d, N, nu, GRID_005)                      # archive: gamma0/N (legacy)
    ga, ia = A.invert(pa)
    pn = inverse.Problem(ev, d, N, nu, GRID_005, cfg=INV_V1_BASELINE, legacy_baseline=True)
    gn, inn = inverse.invert(pn)
    diff = float(np.max(np.abs(ga - gn)))
    assert diff < 1e-10, diff
    return diff


def test_I3_failure_becomes_abstention():
    ev, d, N, lam, HN = _case(N=400)
    pr = inverse.Problem(ev, d, N, laws.plugin_multi(lam), GRID_005, cfg=INV_SCALEFREE)
    orig = inverse._run_slsqp
    inverse._run_slsqp = lambda p, g0, mi, ft: (g0, dict(nit=0, success=False, msg="forced", grad_resid=0.0,
                                                            simplex_resid=0.0, raw_min=0.0, raw_sum=1.0))
    try:
        g, rec = inverse.invert_or_abstain(pr)
    finally:
        inverse._run_slsqp = orig
    assert g is None and rec["abstain"] and rec["reason"] == "solver_nonconvergence", rec
    assert len(rec["detail"]["attempts"]) == 2          # original + the one frozen retry
    return rec["reason"]


if __name__ == "__main__":
    failed = []
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        try:
            print(f"[PASS] {name}: {fn()}", flush=True)
        except Exception as exc:                                      # noqa: BLE001
            failed.append(name); print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
    print("ALL PASS" if not failed else f"SOME FAILED: {failed}")
    sys.exit(1 if failed else 0)
