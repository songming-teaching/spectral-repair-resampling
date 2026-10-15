"""srlab.runbudget -- the EXPERIMENT-level wall-clock budget (the second of two fuses).

THREE GATES, THREE LAYERS, NEVER MERGED (v28 §11.3 / v32 §12.4 / a recorded design decision stage F):

    kernel    T_tail = 60 s          one `law_quantiles` call is bounded        srlab/tail.py
    experiment  total wall clock     one experiment run is bounded              <- this module
    silence     T_silence            no checkpoint for this long => stop        (caller / night runner)

`srlab.tail` answers "may this single solve keep going?".  This module answers "may this RUN start, and
may it keep going?" -- a question a kernel cannot ask, because it does not know how many calls the run
will make.

WHAT GOES WRONG WITHOUT IT (a recorded design decision stage F).  Two failures, both already seen in this project:

  1. A run that dies at hour six leaves PARTIAL data with no record that it was partial.  The
     night-run report had to reconstruct, by hand, which cases had finished.  A total-budget
     exhaustion must therefore write a partial manifest -- status, which cells completed, and
     whether the run may be resumed -- so that partial output is never indistinguishable from
     complete output.
  2. The two gates can CONTRADICT each other before the run starts.  If the per-call budget times
     the expected number of calls exceeds the total budget, the run is guaranteed to be cut off
     mid-way no matter how well it behaves -- and today nothing notices until it happens.  The
     static check below refuses to start such a configuration.

THE BUDGET IS CALIBRATED, NOT GUESSED.  Per v28 §11.3 the total is `pilot x 2`: measure a small
pilot, extrapolate, double it.  `plan()` encodes that; it does not accept a hand-typed total without
a stated pilot.
"""
import time
from dataclasses import dataclass, field, asdict


class RunBudgetExceeded(RuntimeError):
    """The experiment's total wall clock ran out.

    NOT a `ConvergenceError`: this is not a statement about a solve, it is a statement about the run.
    A kernel that runs out of time raises `TailBudgetExceeded` and the experiment layer converts that
    into a registered abstention (M21) and continues; the RUN running out of time is the end of the
    run, and the caller writes a partial manifest.  Merging the two would either let one bad solve
    end a campaign (forbidden) or let a run overrun without limit (the thing this exists to stop)."""

    def __init__(self, msg, partial):
        super().__init__(msg)
        self.partial = partial


@dataclass
class RunBudget:
    """Wall-clock budget for one experiment run.

        rb = RunBudget.from_pilot(seconds=90.0, cases=6, expected_calls=660, what="EXP-3")
        rb.check_start()                     # refuses a configuration the two gates contradict
        for cell in cells:
            rb.check()                       # raises RunBudgetExceeded when out of time
            ...work...
            rb.done(cell)                    # records completion, so a partial run is legible
    """
    seconds: float = 0.0
    what: str = "run"
    expected_calls: int = 0
    tail_seconds: float = 60.0
    expected_seconds: float = 0.0          # pilot-extrapolated HEALTHY cost (not the worst case)
    completed: list = field(default_factory=list)
    total_cells: int = 0
    _t0: float = 0.0
    _started: bool = False

    # ---- construction --------------------------------------------------------------------------
    @classmethod
    def from_pilot(cls, seconds=None, cases=1, pilot_seconds=0.0, pilot_cases=1, tail_seconds=60.0,
                   expected_calls=0, total_cells=0, what="run", safety=2.0):
        """v28 §11.3: total = extrapolated-from-pilot x safety (default 2).

        The pilot must be stated: a total typed in from nowhere is exactly the thing the recorded decision
        forbids ("do not hand-pick the total budget").  `cases / pilot_cases` scales the pilot to the full run.  The
        extrapolated value is kept as `expected_seconds`, which is what the consistency check
        compares against -- NOT the tail ceiling (see `contradiction()`)."""
        if pilot_seconds <= 0 or pilot_cases <= 0:
            raise ValueError("a pilot must be measured: pilot_seconds and pilot_cases must be positive")
        expected = (pilot_seconds / pilot_cases) * max(1, cases)
        total = seconds if seconds is not None else expected * safety
        return cls(seconds=total, what=what, expected_calls=expected_calls, tail_seconds=tail_seconds,
                   expected_seconds=expected, total_cells=total_cells)

    # ---- the static consistency check ----------------------------------------------------------
    def contradiction(self):
        """None if the configuration can terminate, else the reason it cannot.

        WHAT THIS DOES *NOT* CHECK, and why (found 2026-10-03, first version of this file).  The
        literal reading of the recorded decision -- refuse when `T_tail x expected_calls > total` -- refuses
        essentially EVERY real configuration, so it was wrong as stated.  T_tail is a per-call
        CEILING (60 s), while a healthy call costs 0.3-3 s; multiplying the ceiling by the call count
        and comparing it to the budget is a category error, and for EXP-3 it fires as
        `60 s x 660 = 39 600 s > 19 800 s` -- a configuration that is entirely fine.  The worst case
        exceeding the total is in fact NORMAL and INTENDED: when many calls misbehave, the total
        budget is the backstop that bounds the run.  That ratio is reported below as information.

        WHAT IT DOES CHECK is the two ways the layers can genuinely be inverted:
          (a) T_tail alone exceeds the total -- one call can eat the whole run;
          (b) the expected HEALTHY cost (extrapolated from the pilot) exceeds the total -- the run
              cannot finish even if nothing goes wrong."""
        if not self.seconds:
            return None
        if self.tail_seconds > self.seconds:
            return ("T_tail %.0f s exceeds the total budget %.0f s -- one kernel call can consume the "
                    "whole run; the layers are inverted" % (self.tail_seconds, self.seconds))
        if self.expected_seconds and self.expected_seconds > self.seconds:
            return ("the pilot-extrapolated healthy cost %.0f s exceeds the total budget %.0f s -- the "
                    "run cannot finish even if nothing goes wrong"
                    % (self.expected_seconds, self.seconds))
        return None

    def worst_case_ratio(self):
        """(T_tail x calls) / total, or None.  Reported, never a refusal -- see `contradiction()`."""
        if not (self.seconds and self.expected_calls):
            return None
        return self.tail_seconds * self.expected_calls / self.seconds

    def check_start(self):
        """Refuse to start a configuration that cannot terminate."""
        why = self.contradiction()
        if why:
            raise ValueError("[run budget] " + why)
        self._t0 = time.monotonic()
        self._started = True
        return self

    # ---- the clock -----------------------------------------------------------------------------
    def used(self):
        return time.monotonic() - self._t0 if self._started else 0.0

    def remaining(self):
        return self.seconds - self.used()

    def pct(self):
        return 100.0 * self.used() / self.seconds if self.seconds else 0.0

    def eta(self):
        """Projected total seconds from the completed fraction so far, or None if nothing is known.

        This is the number that makes a budget useful DURING a run rather than only after it: an ETA
        past the total is the signal to stop cleanly instead of being killed."""
        if not self.completed or not self.total_cells:
            return None
        frac = len(self.completed) / float(self.total_cells)
        if frac <= 0:
            return None
        return self.used() / frac

    def check(self):
        """Raise when the total budget is gone.  Call once per unit of work."""
        if self._started and self.used() >= self.seconds:
            raise RunBudgetExceeded(
                "[run budget] %s: total wall clock %.0f s exhausted after %d/%d cell(s)"
                % (self.what, self.seconds, len(self.completed), self.total_cells),
                self.partial(status="budget_exceeded"))

    def done(self, cell):
        """Record a completed cell.  A partial run must be able to say what finished."""
        self.completed.append(cell)

    # ---- the partial manifest ------------------------------------------------------------------
    def partial(self, status="running"):
        """What a killed or exhausted run writes, so partial output is never mistaken for complete.

        `resumable` is True while the run stopped for a reason that does not invalidate the cells
        already done (a budget, a wall clock) -- the caller resumes and skips `completed`."""
        return dict(what=self.what, status=status, seconds=self.seconds, used_s=round(self.used(), 3),
                    pct_used=round(self.pct(), 1), total_cells=self.total_cells,
                    completed=[str(c) for c in self.completed],
                    expected_calls=self.expected_calls, tail_seconds=self.tail_seconds,
                    resumable=status == "budget_exceeded")

    def manifest_block(self):
        """What every manifest records (a recorded design decision stage F: usage percent and an ETA projection)."""
        eta = self.eta()
        return dict(what=self.what, budget_s=self.seconds, used_s=round(self.used(), 3),
                    pct_used=round(self.pct(), 1), total_cells=self.total_cells,
                    completed=len(self.completed), eta_s=None if eta is None else round(eta, 1),
                    status="budget_exceeded" if (self._started and self.used() >= self.seconds) else "running")


def pilot_total(pilot_seconds, pilot_cases, cases, safety=2.0):
    """The calibrated total, as a plain number (v28 §11.3).  Exposed for callers that only need it."""
    return (pilot_seconds / max(1, pilot_cases)) * max(1, cases) * safety


def drive(budget, pairs, out_dir):
    """Drive an experiment's task loop under its wall-clock budget -- the stage-F wiring.

    All four experiment entries share one shape: build a task list, dispatch through a Pool, and
    accumulate rows as results arrive.  This consumes that stream so the budget is enforced in ONE
    place rather than four, and so a run that runs out of time WRITES DOWN what it managed instead of
    vanishing (a recorded design decision stage F: "if it dies, it dies" is not acceptable for a run that produced partial data).

        for label, rows in drive(rb, pool.imap_unordered(job_keyed, indexed_tasks), a.out):
            allrows += rows
        status = budget_status(rb)          # "complete" | "budget_exceeded" | "untracked"

    `pairs` must be a LAZY iterable of `(label, rows)` -- pass `pool.imap_unordered(...)` directly,
    never a materialised list.  An earlier version of this function took `tasks` and a `dispatch`
    callable and invoked it one task at a time; that silently serialised the experiment's Pool and
    would have made every window run slower for no reason.  Consuming the stream keeps the
    parallelism exactly where the caller put it.

    The budget is checked BETWEEN results, not inside them: a single task's own time is bounded by
    the kernel gate (T_tail, srlab/tail.py), which is the layer that can actually stop a call.  This
    layer stops the RUN.

    On exhaustion: stops cleanly (no exception into the caller's accumulator), writes
    `<out_dir>/budget.json` (RunBudget.partial) so the partial record ships with the run's outputs and
    reaches the manifest via tools/manifest.py::collect_budget, and returns.  The caller checks
    `budget_status` and exits non-zero -- a partial run must not look like a complete one.

    With `budget is None` this is a plain pass-through: an experiment that declares no budget is not
    silently given one, and its behaviour does not change by a single row.
    """
    if budget is not None:
        budget.check_start()
    it = iter(pairs)
    while True:
        if budget is not None:
            try:
                budget.check()
            except RunBudgetExceeded:
                _write_partial(budget, out_dir, status="budget_exceeded")
                return
        try:
            label, rows = next(it)
        except StopIteration:
            return
        if budget is not None:
            budget.done(label)
        yield label, rows


def _write_partial(budget, out_dir, status="budget_exceeded"):
    """Persist the partial record where the run's outputs live, so it is not lost with the process.

    `status` must be the REASON the run stopped, and it defaults to the only one that calls this
    path today.  Leaving it at the generic default was a real defect caught by the interception
    proof (tests/test_runbudget.py): the run stopped correctly but wrote `status: "running"` and
    `resumable: false`, i.e. a partial record that denied being partial -- the exact failure the
    record exists to prevent."""
    import io
    import json
    import os
    try:
        os.makedirs(out_dir, exist_ok=True)
        with io.open(os.path.join(out_dir, "budget.json"), "w", encoding="utf-8") as fh:
            json.dump(budget.partial(status=status), fh, ensure_ascii=False, indent=1)
    except OSError:
        pass                                    # a run that cannot write its partial record still stops


def budget_status(budget):
    """"untracked" when no budget was declared, else whether the run finished or ran out."""
    if budget is None:
        return "untracked"
    return "budget_exceeded" if budget.used() >= budget.seconds else "complete"

