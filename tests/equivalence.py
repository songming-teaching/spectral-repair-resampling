"""GPU/CPU equivalence harness (M38, spec section III, item 8).

The GPU path may only replace the CPU path after proving equivalence ON THE SAME ARGUMENTS.  The
tolerance is the B-layer one from the spec (fixed grid <= 1e-6 relative); the per-point charter
checks (I5 residual, I4 branch) are additional and independent.

What is compared, per case:
    e            the fixed point (complex128)         <- the quantity everything else derives from
    m            the Stieltjes value                  <- what the callers consume
    resid        per-point |e - F(e)|                 <- the charter's own measure
    iterations   per-point iteration counts           <- a large gap means a different path

Two levels of comparison are offered:
    compare_arrays   numeric equivalence at a declared tolerance
    compare_exact    bitwise, for the seeded/CPU-side quantities that MUST NOT move (R-1: random
                     streams are generated on the CPU; this harness asserts that claim)

A failed equivalence is a hard error, never a warning: the fork's whole premise is that the GPU
reproduces the frozen reference to within a declared tolerance.
"""
import numpy as np

from .invariants import ConvergenceError, require


def compare_arrays(a, b, tol=1e-6, what="arrays"):
    """Relative infinity-norm comparison.  Returns (ok, info); raises nothing so callers can report."""
    a = np.asarray(a); b = np.asarray(b)
    require(a.shape == b.shape, "EQ", f"{what}: shape {a.shape} != {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    scale = max(float(np.max(np.abs(a[finite]))) if finite.any() else 1.0, 1e-300)
    absdiff = np.where(finite, np.abs(a - b), np.inf)
    max_abs = float(np.max(absdiff))
    max_rel = max_abs / scale
    info = dict(what=what, tol=tol, max_abs=max_abs, max_rel=max_rel, scale=scale,
                n=int(a.size), n_nonfinite=int((~finite).sum()),
                max_index=[int(i) for i in np.unravel_index(int(np.argmax(absdiff)), a.shape)] if a.size else [])
    return bool(max_rel <= tol), info


def require_equivalent(a, b, tol=1e-6, what="arrays"):
    ok, info = compare_arrays(a, b, tol, what)
    if not ok:
        raise ConvergenceError(
            f"[EQ] {what}: GPU/CPU differ beyond tolerance {tol:.0e} "
            f"(max_rel {info['max_rel']:.3e} at {info['max_index']}, scale {info['scale']:.3e})")
    return info


def compare_exact(a, b, what="arrays"):
    """Bitwise.  Used for the CPU-side quantities that must not move: seeds, weights, counts."""
    a = np.asarray(a); b = np.asarray(b)
    same = a.shape == b.shape and np.array_equal(a, b)
    info = dict(what=what, n=int(a.size), bitwise=bool(same),
                n_diff=int(np.sum(a != b)) if a.shape == b.shape else -1)
    return bool(same), info


def require_exact(a, b, what="arrays"):
    ok, info = compare_exact(a, b, what)
    if not ok:
        raise ConvergenceError(f"[EQ] {what}: not bitwise identical ({info['n_diff']} differing elements) "
                               f"-- this quantity must not move (R-1: random streams live on the CPU)")
    return info


def report(rows, title):
    """Print a fixed-width equivalence table and return the overall verdict."""
    ok_all = True
    print(f"\n=== {title} ===")
    print(f"{'case':<34}{'what':<10}{'max_rel':>12}{'max_abs':>12}{'tol':>10}  verdict")
    for r in rows:
        ok_all &= bool(r["ok"])
        print(f"{r['case']:<34}{r['what']:<10}{r['max_rel']:>12.3e}{r['max_abs']:>12.3e}"
              f"{r['tol']:>10.0e}  {'OK' if r['ok'] else 'FAIL'}")
    print("EQUIVALENT" if ok_all else "NOT EQUIVALENT")
    return ok_all
