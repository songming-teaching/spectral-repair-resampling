"""srlab.tail -- tail-latency discipline: wall-clock budgets and the typed escalation ladder.

THE PROBLEM THIS SOLVES.  Historically, work measured in minutes repeatedly ran for hours:

    S7 c=0.5 reference rerun      est. 6-8 min    actual >200 min, abandoned after 3.5 h
    M18 reference dual-run                                       4.0 h
    single delta_fb after M17     <1 s before   106-681 s after
    D20 timing (24 cases)         minutes        est. 10-25 h

Mechanism: a fixed-point solve runs `continuation levels x iteration cap` = 5 x 4000 = 20000 steps over
the WHOLE grid array, and under the historical GLOBAL convergence criterion every point waited for the
slowest one.  Measured single step at nz=400k: 596 ms (CPU, 1 thread) -> worst case 3.3 h.

THE THREE LEVERS (all three are required; any one alone leaves the tail unbounded):

    1. per-point criterion + active set    the slowest point stops dragging the others   [M35]
    2. GPU inner kernel for the memory-bound iteration    596 ms -> 32.5 ms per step     [M38]
    3. WALL-CLOCK BUDGET + typed escalation + typed event        <- this module

Levers 1+3 give BOUNDEDNESS; lever 2 gives speed.  With only 2, the worst case becomes "an unknown
11 minutes" instead of "an unknown 3.3 hours" -- better, still unbounded.

CONTRACT.  Every potentially hour-scale call is wrapped in a `TailBudget`.  On exhaustion the kernel
STOPS (never silently truncates), records what it managed, and raises `TailBudgetExceeded`, which is a
`ConvergenceError` and therefore a typed event: the experiment layer maps it to a registered
abstention with a reason code (M21).  A campaign is never aborted and a value is never returned
unconverged -- the two things the charter forbids.

Every attempt appends an `Attempt` record: level, wall seconds, iterations, active-set size, points
still unconverged, and the outcome.  These land in the manifest (M37/D25: whatever influenced the
numbers is recorded).
"""
import time
from dataclasses import dataclass, field, asdict

from .invariants import ConvergenceError

# The default tail budget for a SINGLE kernel invocation.  Chosen so that a full experiment cannot be
# derailed by one bad point: EXP-3 has 660 supervisor steps, so 60 s/call caps the pathological cost at
# 660 x 60 s = 11 h worst-case-if-every-step-exhausted, while a healthy step costs ~0.3-3 s.
DEFAULT_TAIL_SECONDS = 60.0

# Sub-budgets: lever 1 is expected to resolve almost everything (measured: stragglers were 0 of
# 400 000 points on healthy cases, 6 and 378 on the two known-bad D24 steps), so the ladder gives the
# cheap level most of the budget and keeps a reserve for the expensive ones.
LEVEL_BUDGET_FRACTION = (0.5, 0.3, 0.2)


class TailBudgetExceeded(ConvergenceError):
    """The wall-clock budget for one kernel invocation ran out.  A TYPED event (M21/M35): the
    experiment layer records an abstention with a reason code; it never aborts a campaign and never
    returns an unconverged value.

    Carries its OWN registered reason code, distinct from `solver_nonconvergence` (recorded decision,
.  O-1's whole point is that the budget is the ONLY source of boundedness, and
    a ledger that cannot tell "the budget fired" from "the escalation ladder ran out" cannot show that
    lever working.  The two are orthogonal dimensions: this code names the MECHANISM, the call site
    prefixes the QUANTITY (e.g. "dfb_tail_budget_exceeded")."""

    reason_code = "tail_budget_exceeded"

    def __init__(self, msg, info):
        super().__init__(msg)
        self.info = info


@dataclass
class Attempt:
    """One rung of the ladder.  Everything that could influence the numbers is recorded."""
    level: int
    label: str
    wall_s: float = 0.0
    iterations: int = 0
    points: int = 0
    entered_unconverged: int = 0
    left_unconverged: int = 0
    max_resid: float = 0.0
    path: str = "cpu"                 # "cpu" | "gpu" | "gpu+cpu-fallback"
    chunk: int = 0                    # GPU only: z-points per submission
    outcome: str = "pending"          # converged | partial | budget_exhausted | error
    note: str = ""


@dataclass
class Budget:
    """Wall-clock budget for one kernel invocation.

        with Budget(60.0, "law_quantiles", attempts) as b:
            for level in ...:
                b.start_level(level, "pointwise")
                while ...:
                    b.check()          # raises TailBudgetExceeded when out of time
    """
    seconds: float = DEFAULT_TAIL_SECONDS
    what: str = "kernel"
    attempts: list = field(default_factory=list)
    splits: tuple = LEVEL_BUDGET_FRACTION
    _t0: float = 0.0
    _level: Attempt | None = None
    _level_t0: float = 0.0
    _level_budget: float = 0.0

    def __enter__(self):
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._level is not None:
            self._level.wall_s = time.monotonic() - self._level_t0
            if self._level.outcome == "pending":
                self._level.outcome = "error" if exc_type else "converged"
        return False

    # ---- clock -------------------------------------------------------------------------------
    def used(self):
        return time.monotonic() - self._t0

    def remaining(self):
        return self.seconds - self.used()

    def check(self):
        """Raise when the TOTAL budget is gone.  Call once per iteration."""
        if self.used() >= self.seconds:
            self._finish("budget_exhausted")
            raise TailBudgetExceeded(
                f"[T_tail] {self.what}: wall-clock budget {self.seconds:.0f} s exhausted "
                f"after {len(self.attempts)} level(s)",
                dict(what=self.what, budget_s=self.seconds, used_s=self.used(),
                     attempts=[asdict(a) for a in self.attempts]))

    # ---- the ladder --------------------------------------------------------------------------
    def start_level(self, level, label):
        """Open a rung.  Its own share of the budget is what remains, split by LEVEL_BUDGET_FRACTION.
        A level that overruns its share is cut off, but the later levels keep their reserve."""
        if self._level is not None:
            self._level.wall_s = time.monotonic() - self._level_t0
            if self._level.outcome == "pending":
                self._level.outcome = "partial"
        share = self.splits[min(level, len(self.splits) - 1)]
        self._level_budget = share * self.seconds
        self._level_t0 = time.monotonic()
        a = Attempt(level=level, label=label)
        self.attempts.append(a)
        self._level = a
        return a

    def level_expired(self):
        """True when THIS rung has used its share (cut it off and move up the ladder)."""
        return self._level is not None and (time.monotonic() - self._level_t0) >= self._level_budget

    def note(self, **kw):
        if self._level is not None:
            for k, v in kw.items():
                setattr(self._level, k, v)

    def _finish(self, outcome):
        if self._level is not None:
            self._level.wall_s = time.monotonic() - self._level_t0
            if self._level.outcome == "pending":
                self._level.outcome = outcome
            self._level = None

    def finish(self, outcome="converged"):
        self._finish(outcome)

    # ---- reporting ---------------------------------------------------------------------------
    def manifest_block(self):
        """What the manifest records: budget, use, and the ladder as executed."""
        return dict(what=self.what, budget_s=self.seconds, used_s=round(self.used(), 3),
                    levels=[asdict(a) for a in self.attempts],
                    exhausted=any(a.outcome == "budget_exhausted" for a in self.attempts))


def budget_for(n_calls, total_seconds, floor=DEFAULT_TAIL_SECONDS):
    """Per-call budget so that n_calls invocations fit in total_seconds, never below `floor`."""
    return max(floor, total_seconds / max(1, n_calls))
