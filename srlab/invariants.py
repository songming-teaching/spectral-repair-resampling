"""srlab.invariants -- the platform charter (the project records section 0).

These are not regression tolerances.  A violated invariant means a numerical result is not
trustworthy; the computation stops.  No code path may silently fall back.
"""
import numpy as np


class InvariantError(RuntimeError):
    """A charter invariant (I1-I8) is violated."""


class ConvergenceError(InvariantError):
    """I5: a fixed point did not converge on every requested point.

    O-1(c) / M38: every subclass of this class is a TYPED experiment-layer event.  The experiment
    layer must catch THIS class, never one of its subclasses.  `SolverEscalationFailure` and
    `TailBudgetExceeded` are SIBLINGS, and the plain `ConvergenceError` that `check_converged` raises
    from inside `Problem.model` is the PARENT -- three separate ways for a convergence event to leave
    the solver, and a narrow `except SolverEscalationFailure` let two of them escape and abort a
    campaign, which is exactly what the charter forbids.

    Each subclass declares the registered `reason_code` it maps to; `abstention_reason` reads it so a
    new subclass cannot silently escape.  The code names the MECHANISM, the call site names the
    QUANTITY (srlab.supervisor builds e.g. "dfb_" + code)."""

    #: registered abstention reason code (M21).  Subclasses override.
    reason_code = "solver_nonconvergence"


class SolverEscalationFailure(ConvergenceError):
    """M35: the full escalation sequence (pointwise -> per-point dense continuation -> single-point
    iteration-cap upgrade) left points unconverged.  A TYPED event: the experiment layer maps it to a
    registered abstention (M21 layering), it never aborts a campaign and is never silenced.

    Carries the same reason code as a plain non-convergence: to the experiment layer both mean "the
    solve did not converge", and the registered vocabulary (M21/M35) has one name for it.  The
    escalation detail stays in `info`."""

    reason_code = "solver_nonconvergence"

    def __init__(self, msg, info):
        super().__init__(msg)
        self.info = info


#: THE registered abstention vocabulary (M21/M35/M38).  One place, so a new subclass cannot invent a
#: code that nobody registered -- the test suite asserts every class's `reason_code` is in here.
#:
#:   solver_nonconvergence   the fixed point did not converge: either the M35 escalation ladder
#:                           exhausted, or `check_converged` raised the base class from inside the
#:                           objective.  One name, because to the experiment layer both mean the same
#:                           thing; the escalation detail stays in `info`.
#:   tail_budget_exceeded    the wall-clock budget ran out (M38).  Kept SEPARATE from the above by the
#:                           2026-10-03 recorded decision: the budget is the only source of boundedness, and a
#:                           ledger that cannot distinguish it cannot demonstrate that lever working.
#:                           Orthogonal to the call-site PREFIX (e.g. "dfb_" + code).
REGISTERED_REASON_CODES = frozenset({"solver_nonconvergence", "tail_budget_exceeded"})


def abstention_reason(exc):
    """O-1(c): the registered reason code for a typed experiment-layer event.

    Callers catch `ConvergenceError` and tag the abstention with `quantity + "_" + this`, so a new
    subclass cannot silently escape a narrow `except` clause.  Falls back to the base code for a bare
    `ConvergenceError` (what `check_converged` raises inside `Problem.model`).

    Raises if a class declares a code outside `REGISTERED_REASON_CODES`: an unregistered code entering
    an abstention record is a vocabulary change that never went through the preregistration chain."""
    code = getattr(exc, "reason_code", None) or ConvergenceError.reason_code
    if code not in REGISTERED_REASON_CODES:
        raise InvariantError(f"[M21] abstention_reason: {type(exc).__name__} declares unregistered "
                             f"reason code {code!r}; registered = {sorted(REGISTERED_REASON_CODES)}")
    return code


def require(cond, code, msg):
    if not cond:
        raise InvariantError(f"[{code}] {msg}")


def check_count_law(p, t=None, tol=1e-9, what="count law"):
    """I1: probability vector with exact unit mean (m1(nu) = 1)."""
    p = np.asarray(p, float)
    t = np.arange(len(p), dtype=float) if t is None else np.asarray(t, float)
    require(np.all(p >= 0) and np.all(np.isfinite(p)), "I1", f"{what}: negative or non-finite mass")
    require(abs(p.sum() - 1.0) < tol, "I1", f"{what}: total mass {p.sum():.3e} != 1")
    m1 = float(p @ t)
    require(abs(m1 - 1.0) < tol, "I1", f"{what}: m1 = {m1:.12f} != 1")
    return m1


def check_rank_identity(p0_emp, n_active, d, centred=True):
    """I2: zero atom of the (centred) resampled ESD equals the count-rank identity."""
    rank = min(d, n_active - (1 if centred else 0))
    expected = max(0.0, 1.0 - rank / d)
    require(abs(p0_emp - expected) < 0.5 / d, "I2",
            f"zero atom {p0_emp:.6f} != count-rank identity {expected:.6f} (n_active={n_active}, d={d})")
    return expected


def check_branch(e, m=None, what="solution"):
    """I4: Stieltjes branch on every returned point."""
    require(np.all(np.isfinite(e)), "I4", f"{what}: non-finite e")
    require(np.all(e.imag > 0), "I4", f"{what}: Im e <= 0 at {int(np.sum(e.imag <= 0))} points")
    if m is not None:
        require(np.all(np.isfinite(m)) and np.all(m.imag > 0), "I4", f"{what}: Im m <= 0 or non-finite")


def check_converged(resid, accept, what="fixed point"):
    """I5: every point converged (per-point residual |e - F(e)|)."""
    bad = ~(resid < accept)
    if np.any(bad):
        raise ConvergenceError(f"[I5] {what}: {int(bad.sum())}/{len(resid)} points unconverged "
                               f"(max resid {float(np.nanmax(resid)):.3e} >= {accept:.1e})")


def normalise(p, what, tol=1e-12):
    """I9 (permanent sentinel, M34): a distribution may only be renormalised after its PRE-normalisation
    mass gap has been measured, reported and found below `tol`.  `p / p.sum()` without this is forbidden
    in srlab (tools/lint_normalisation.py enforces it): it silently repairs a wrong integral -- the D23
    residual-oracle law had exact unit mass after normalisation and m1 = 1.00019.
    Returns (normalised p, gap) where gap = 1 - sum(p) BEFORE normalisation."""
    p = np.asarray(p, float)
    tot = float(p.sum())
    gap = 1.0 - tot
    require(np.isfinite(tot) and tot > 0, "I9", f"{what}: total mass {tot!r} not positive / finite")
    require(abs(gap) <= tol, "I9", f"{what}: pre-normalisation mass gap {gap:+.3e} exceeds {tol:.0e}")
    return p / tot, gap
