"""Count-law tests (M34, D23 fix + I9 sentinel).

  L1  every guarded oracle law has m1 = 1 to 1e-12 (the charter's I1, no longer 1e-7);
  L2  the exact residual construction leaves a pre-normalisation mass gap < 1e-14 -- the integral is
      right BEFORE any renormalisation (the I9 sentinel's point);
  L3  where the legacy quadrature was accurate (alpha = 1, 0.3) the exact law agrees with it to 1e-13
      -- an independent cross-check of the new construction;
  L4  the legacy law is refused by the guarded path at alpha = 10 (D23) and is reachable only through
      oracle_legacy(), which reports the error instead of hiding it;
  L5  normalise() raises on a mass gap above tolerance and is arithmetically identical to p / p.sum().
Run:  python tests/test_laws.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srlab import laws                                       # noqa: E402
from srlab.invariants import InvariantError, normalise       # noqa: E402

ALPHAS = (10.0, 3.0, 1.0, 0.5, 0.3)


def test_L1_unit_mean():
    worst = 0.0
    for sch in ("multi", "res"):
        for a in ALPHAS:
            p = laws.oracle(sch, a)
            worst = max(worst, abs(float(p @ np.arange(len(p))) - 1.0))
    assert worst < 1e-12, worst
    return f"max |m1 - 1| = {worst:.1e}"


def test_L2_prenormalisation_gap():
    worst = max(abs(laws._oracle_res_gamma_exact(a)[1]) for a in ALPHAS)
    assert worst < 1e-14, worst
    return f"max |gap| = {worst:.1e}"


def test_L3_agrees_with_legacy_where_legacy_was_right():
    worst = 0.0
    for a in (1.0, 0.3):
        new = laws.oracle("res", a)
        old, err = laws.oracle_legacy("res", a)
        n = min(len(new), len(old))
        worst = max(worst, float(np.max(np.abs(new[:n] - old[:n]))))
    assert worst < 1e-13, worst
    return f"max |new - legacy| = {worst:.1e}"


def test_L4_legacy_is_refused_and_reported():
    old, err = laws.oracle_legacy("res", 10.0)
    assert abs(err - 1.897e-4) < 1e-6, err
    raw = laws._raw_oracle_res_gamma(10.0)
    try:
        laws._checked(laws.trim_pmf(raw), "legacy res Gamma(10)", tol=1e-12)
    except InvariantError:
        return f"legacy m1 error {err:+.3e} reported; guarded path refuses it"
    raise AssertionError("legacy law passed the guard")


def test_L5_normalise():
    p = np.array([0.2, 0.3, 0.5000000001])
    q, gap = normalise(p, "t", tol=1e-9)
    assert np.array_equal(q, p / p.sum()) and abs(gap + 1e-10) < 1e-15
    try:
        normalise(np.array([0.2, 0.3, 0.49]), "t", tol=1e-9)
    except InvariantError:
        return "identical arithmetic; raises on gap 1e-2"
    raise AssertionError("gap 1e-2 not caught")


if __name__ == "__main__":
    failed = []
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        try:
            print(f"[PASS] {name}: {fn()}", flush=True)
        except Exception as exc:                                  # noqa: BLE001
            failed.append(name); print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
    print("ALL PASS" if not failed else f"SOME FAILED: {failed}")
    sys.exit(1 if failed else 0)
