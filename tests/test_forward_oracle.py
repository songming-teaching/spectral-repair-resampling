"""Oracle equivalence: the migrated platform against the FROZEN archive implementation.

P1 acceptance (the project records): forward map <= 1e-12, quantiles <= 1e-5.
The reference is `archive/code_2026-09-27/srcore.py` -- loaded by path under a private module name
so that its `import` statements (which assume a flat directory) do not collide with the package.

Covered:
  * `stieltjes` / `stieltjes_dense` against archive `stieltjes` on the same arguments;
  * `forward.law_quantiles` (fast) against archive `law_quantiles` (M18 fast version) AND against
    `law_quantiles_ref` (M17 reference solver) -- the M18 acceptance quantity;
  * the I6 cap bookkeeping records the historical behaviour (M26: report, never silent).

Run:  python tests/test_forward_oracle.py
"""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "reference")
sys.path.insert(0, ROOT)

from srlab import forward, solver, laws          # noqa: E402
from srlab.config import INV_SCALEFREE, QUANT    # noqa: E402

U = (np.arange(20001) + 0.5) / 20001


def _load_archive():
    spec = importlib.util.spec_from_file_location("_arch_srcore", os.path.join(ARCHIVE, "srcore.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_arch_srcore"] = mod
    spec.loader.exec_module(mod)
    return mod


ARCH = _load_archive()


def _cases(full=False):
    """Oracle cases.

    Default: the same arguments with an EXPLICIT window (no growth) and a light count law, so the
    migrated arithmetic is compared without paying the window-growth cost.  With SR_FULL=1 the
    full-growth cases from the M17 acceptance grid run too -- they sit on the 4e5 cap (M26) and
    take minutes each; equivalence there is already recorded in the M17/M18 acceptance logs."""
    # A FIXED window (x_hi = 2.0) keeps the grid to ~4e4 points: the growth loop is what makes the
    # capped cases slow, and it is not what is being tested here.  Both the new and the archive
    # implementations receive identical arguments, so this is an arithmetic equivalence check.
    for c in (0.15, 0.75, 1.25):
        nu = laws.plugin_multi(np.ones(10))
        t = np.arange(len(nu), dtype=float)
        yield f"poisson(1)/c={c}", [1.0], np.array([1.0]), t, nu, c, 2.0
    if full:
        nu = laws.oracle("multi", 0.3)
        t = np.arange(len(nu), dtype=float)
        for c in (0.15, 0.5, 0.75):
            yield f"H0/NB(0.3)/c={c}", [0.5, 2.5], np.array([0.7, 0.3]), t, nu, c, None


def test_stieltjes_equivalence():
    """solver.stieltjes must equal archive stieltjes to the charter tolerance."""
    worst = 0.0
    for c in (0.15, 0.75, 1.25):
        nu = laws.plugin_multi(np.ones(10))
        t = np.arange(len(nu), dtype=float)
        z = np.linspace(0.1, 5.0, 60) + 0.01j
        m_new = solver.stieltjes(z, [1.0], [1.0], t, nu, c)[0]
        m_old = ARCH.stieltjes(z, [1.0], [1.0], t, nu, c)[0]
        worst = max(worst, float(np.max(np.abs(m_new - m_old))))
    assert worst <= 1e-12, worst
    return worst


def test_quantiles_vs_archive_fast_and_ref():
    """forward.law_quantiles vs both archive versions; M18 acceptance tolerance 1e-5.

    THREE QUESTIONS, THREE TOLERANCES.  They were previously conflated into one bitwise assertion,
    which conflated two different things and hid a stale expectation (see below).

      (A) MIGRATION -- does the current implementation reproduce the archive arithmetic EXACTLY?
          The archive ran the historical GLOBAL convergence criterion, which is what
          `legacy_unchecked=True` selects; with that flag the comparison is bitwise (measured 0.0).
          This is the real migration guarantee: the same nominal algorithm gives the same numbers.
      (A2) PRODUCTION -- the default path uses M35's POINTWISE criterion, which is a deliberate
          algorithmic change, so it is NOT bitwise against the archive.  Measured 6.99e-12 on the
          H0/NB(0.3)/c=0.75 case (the one that hits the 4e5 cap).  The registered criterion here is
          M18's 1e-5, which it passes by six orders; the deviation is recorded, not asserted away.
      (B) BACKEND -- does the GPU path agree with the CPU one?  Compared on the INTEGRAL layer
          (moments), which is what the paper cites and what the freeze-window criterion uses
          (<=1e-13, recorded decision); the pointwise layer is reported, not asserted.

    WHY THIS TEST WAS BROKEN AND NOBODY NOTICED.  The `worst_fast <= 1e-12` line encoded a PRE-M35
    expectation.  Since M35 changed the default criterion it has been false whenever the full branch
    runs -- but CI never sets SR_FULL, so the full branch has never executed there.  Verified
    2026-10-03 by running the HEAD revision of this file against the current tree: it fails with the
    same 2.14e-12 (GPU) / 6.99e-12 (CPU-pinned), i.e. the breakage predates this session.  The
    overnight queue's SR_FULL task is what exercised it for the first time.

    Cost control: the reference solver iterates the WHOLE 40k-point array, so it is ~100x the fast
    path.  Fast-vs-fast runs on every case; fast-vs-ref (1e-5) runs on the first one only -- a subset
    of the M18 acceptance grid whose full record is in data/m18/verify_M18_ab.json."""
    full = os.environ.get("SR_FULL") == "1"
    worst_legacy = 0.0          # (A)  legacy flag vs archive: must be bitwise
    worst_strict = 0.0          # (A2) production default vs archive: M18 tolerance
    cases = list(_cases(full=full))
    with forward.pin_cpu_backend():
        for tag, tau, g, t, nu, c, x_hi in cases[1:]:
            q_arch = ARCH.law_quantiles(tau, g, t, nu, c, U, x_hi=x_hi)
            q_leg, _ = forward.law_quantiles(tau, g, t, nu, c, U, x_hi=x_hi, legacy_unchecked=True)
            q_str, _ = forward.law_quantiles(tau, g, t, nu, c, U, x_hi=x_hi)
            worst_legacy = max(worst_legacy, float(np.max(np.abs(q_leg - q_arch))))
            worst_strict = max(worst_strict, float(np.max(np.abs(q_str - q_arch))))

        # fast-vs-ref on a COARSE eta: the reference solver iterates the entire grid array under a
        # global convergence criterion, so its cost explodes with n (M18's whole motivation).  The
        # arithmetic being compared does not depend on which eta is used, so a coarse-grid comparison
        # is the honest, affordable form of this check.  The full-grid record is data/m18/.
        tag, tau, g, t, nu, c, x_hi = cases[0]
        ETA_COARSE = 1e-3
        q_ref = ARCH.law_quantiles_ref(tau, g, t, nu, c, U, x_hi=x_hi, eta_rel=ETA_COARSE)
        q_ref_new = forward.law_quantiles_ref(tau, g, t, nu, c, U, x_hi=x_hi, eta_rel=ETA_COARSE)[0]
        worst_ref = float(np.max(np.abs(q_ref_new - q_ref)))
        assert worst_ref <= 1e-12, worst_ref      # same routine, same solver, must be bitwise
        q_new, info = forward.law_quantiles(tau, g, t, nu, c, U, x_hi=x_hi)
        q_fast_coarse = forward.law_quantiles(tau, g, t, nu, c, U, x_hi=x_hi, eta_rel=ETA_COARSE)[0]
        d_grid = float(np.max(np.abs(q_fast_coarse - q_ref)))
        assert d_grid <= 1e-5, d_grid             # fast vs ref across the grid-resolution change
        assert worst_legacy <= 1e-12, worst_legacy    # (A) migration, CPU-pinned, historical criterion
        assert worst_strict <= 1e-5, worst_strict     # (A2) production default, M18 tolerance
        assert worst_ref <= 1e-5, worst_ref
        assert np.isfinite(info["m1_capture"]) and "cap" in info

    # (B) backend equivalence on the integral layer, on this same case.
    with forward.pin_cpu_backend():
        q_cpu, _ = forward.law_quantiles(tau, g, t, nu, c, U, x_hi=x_hi)
    q_gpu, info_gpu = forward.law_quantiles(tau, g, t, nu, c, U, x_hi=x_hi)
    int_rel, pt = None, None
    if (info_gpu.get("solver") or {}).get("path", "").startswith("gpu"):
        m1_c = float(np.trapezoid(q_cpu, U)); m1_g = float(np.trapezoid(q_gpu, U))
        m2_c = float(np.trapezoid(q_cpu * q_cpu, U)); m2_g = float(np.trapezoid(q_gpu * q_gpu, U))
        int_rel = max(abs(m1_g - m1_c) / max(abs(m1_c), 1e-300), abs(m2_g - m2_c) / max(abs(m2_c), 1e-300))
        pt = float(np.max(np.abs(q_gpu - q_cpu)))
        assert int_rel <= 1e-13, (int_rel, pt, "integral layer is what the paper cites")
    return dict(legacy_vs_archive=worst_legacy, strict_vs_archive=worst_strict,
                ref_vs_ref=worst_ref, gpu_cpu_pointwise=pt, gpu_cpu_integral=int_rel)


def test_i6_bookkeeping_reports():
    """M26: under cap_policy='report' the shortfall is recorded and no number moves."""
    if os.environ.get("SR_FULL") != "1":
        return "skipped (set SR_FULL=1; capped cases are slow)"
    # `t` and `nu` have to be built HERE.  This test used to reference module-level `t`/`nu` that do
    # not exist (the module only defines U), so it raised NameError -- and because CI never sets
    # SR_FULL it was skipped everywhere and the breakage went unnoticed until the overnight run
    # (2026-10-03) exercised the full-suite path.  Built the same way as test_i6_raise_policy below.
    nu = laws.oracle("multi", 0.3)
    t = np.arange(len(nu), dtype=float)
    for c, expect_exceed in ((0.15, False), (0.5, True), (0.75, True), (1.25, True)):
        q, info = forward.law_quantiles([1.0], np.array([1.0]), t, nu, c, U)
        cap = info["cap"]
        assert cap["policy"] == "report"
        if expect_exceed:
            assert cap["exceeded"] and cap["worst_dx_over_eta"] > 1.0, (c, cap)
        else:
            assert not cap["exceeded"], (c, cap)
    return True


def test_i6_raise_policy():
    """M26: cap_policy='raise' turns the same case into a hard I6 failure."""
    nu = laws.oracle("multi", 0.3)
    t = np.arange(len(nu), dtype=float)
    from srlab.invariants import InvariantError
    if os.environ.get("SR_FULL") != "1":
        return "skipped (set SR_FULL=1; capped case is slow)"
    # delta_1 / c = 1.25 / NB(0.3): n_needed / n_cap = 2.68 (M26 probe) -- a case that DOES hit the cap.
    # (An earlier edit used Poisson(1), which never reaches the cap: the test would have failed falsely.)
    try:
        forward.law_quantiles([1.0], np.array([1.0]), t, nu, 1.25, U, cap_policy="raise")
    except InvariantError as exc:
        assert "I6" in str(exc)
        return True
    raise AssertionError("cap_policy='raise' did not raise on a capped case")


def test_legacy_gamma_guard():
    """M19: no experiment path may silently pick up the pre-M19 gamma rule."""
    from srlab.inverse import Problem
    from srlab.config import INV_V1_BASELINE
    from srlab.invariants import InvariantError
    try:
        Problem(np.ones(20), 20, 40, np.array([0.5, 0.5]), np.array([1.0]), cfg=INV_V1_BASELINE)
    except InvariantError as exc:
        assert "legacy_N" in str(exc)
        return True
    raise AssertionError("legacy gamma rule was not guarded")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            res = fn()
            tag = "SKIP" if isinstance(res, str) and res.startswith("skipped") else "PASS"
            print(f"[{tag}] {name}: {res}", flush=True)
        except AssertionError as exc:
            failed.append(name)
            print(f"[FAIL] {name}: {exc}", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            failed.append(name)
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}", flush=True)
    print("ALL PASS" if not failed else f"SOME FAILED: {failed}")
    sys.exit(1 if failed else 0)
