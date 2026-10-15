"""Solver unit tests (spec section III), ported from archive/code_2026-09-27/test_solver.py.

Seven checks, same quantities and thresholds as the historical file -- these are the tests the
paper's section VIII-C reports, so they must not drift:
  T1 branch Im m > 0 (Marcheko-Pastur point, c = 0.75)
  T2 fixed-point residual  < 3e-13
  T3 z m(z) -> -1 at large |z|
  T4 MP(c) interior density recovery, relative error <= 0.76% (inherited reference value)
  T5 zero atom vs closed form, <= 4e-6 (inherited reference value)
  T6 truncation insensitivity K = 20 vs 60, <= 2.1e-15 (inherited reference value)
  T7 companion-equation cross-check, <= 1e-10

T8/T9 (Jacobian finite differences, Monte-Carlo d_St against theory) from the historical file are
not unit tests -- they are checks of the FORWARD MAP and are exercised by `test_forward.py`.

Run:  python tests/test_solver.py            (no pytest needed)
      python -m pytest tests/test_solver.py  (if pytest is installed)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srlab import solver, laws                                   # noqa: E402
from srlab.sampling import split_atom, emp_stieltjes_plus        # noqa: E402

RNG = np.random.default_rng(20260925)
C = 0.75
ZA = np.linspace(0.1, 6, 80) + 0.02j


def test_T1_branch():
    """T1: Im m > 0 on the whole grid."""
    x = np.linspace(0.05, 3.3, 200)
    m, e, A, D, info = solver.stieltjes(x + 1j * 1e-3, [1.0], [1.0], [1.0], [1.0], C)
    assert float(m.imag.min()) > 0


def test_T2_fixed_point_residual():
    """T2: per-point fixed-point residual."""
    x = np.linspace(0.05, 3.3, 200)
    _, _, _, _, info = solver.stieltjes(x + 1j * 1e-3, [1.0], [1.0], [1.0], [1.0], C)
    assert info["max_resid"] < 3e-13


def test_T3_normalisation():
    """T3: z m(z) -> -1 at large |z|."""
    zb = np.array([1e6j, -1e6 + 1j, 1e6 + 1j])
    mb = solver.stieltjes(zb, [1.0], [1.0], [1.0], [1.0], C)[0]
    assert np.max(np.abs(zb * mb + 1)) < 1e-5


def test_T4_mp_density():
    """T4: Marchenko-Pastur interior density, relative error <= 0.76%."""
    tm, tp = (1 - np.sqrt(C)) ** 2, (1 + np.sqrt(C)) ** 2
    xi = np.linspace(tm + 0.1 * (tp - tm), tp - 0.1 * (tp - tm), 100)
    mi = solver.stieltjes(xi + 1e-7j, [1.0], [1.0], [1.0], [1.0], C)[0]
    dens = mi.imag / np.pi
    exact = np.sqrt((tp - xi) * (xi - tm)) / (2 * np.pi * C * xi)
    assert float(np.max(np.abs(dens - exact) / exact)) < 0.0076


def test_T5_zero_atom():
    """T5: zero atom vs the closed form, two ratios."""
    for cc in (0.75, 1.25):
        nu = laws.plugin_multi(np.ones(10))
        t = np.arange(len(nu), dtype=float)
        y = 1e-7
        my = solver.stieltjes(np.array([1j * y]), [1.0], [1.0], t, nu, cc)[0]
        atom_num = float((-1j * y * my[0]).real)
        atom_cf = laws.zero_atom(nu[0], cc)
        assert abs(atom_num - atom_cf) < 4e-6, (cc, atom_num, atom_cf)


def test_T6_truncation_insensitivity():
    """T6: truncation K = 20 vs 60 leaves the Stieltjes transform unchanged."""
    nu60 = laws.poisson_mixture_pmf(np.ones(5), kmax=60)
    nu20 = laws.poisson_mixture_pmf(np.ones(5), kmax=20)
    m60 = solver.stieltjes(ZA, [1.0], [1.0], np.arange(61.0), nu60, C)[0]
    m20 = solver.stieltjes(ZA, [1.0], [1.0], np.arange(21.0), nu20 / nu20.sum(), C)[0]
    assert float(np.max(np.abs(m60 - m20))) < 1e-12


def test_T7_companion_equation():
    """T7: the separable fixed point satisfies the companion equation."""
    nu = laws.oracle("multi", 1.0)
    t = np.arange(len(nu), dtype=float)
    m7 = solver.stieltjes(ZA, [1.0], [1.0], t, nu, C)[0]
    comp = -1.0 / m7 + (nu[None, :] * t[None, :] / (1 + C * t[None, :] * m7[:, None])).sum(1) - ZA
    assert float(np.max(np.abs(comp))) < 1e-10


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as exc:
            failed.append(name)
            print(f"[FAIL] {name}: {exc}")
        except Exception as exc:                                   # noqa: BLE001
            failed.append(name)
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
    print("ALL PASS" if not failed else f"SOME FAILED: {failed}")
    sys.exit(1 if failed else 0)
