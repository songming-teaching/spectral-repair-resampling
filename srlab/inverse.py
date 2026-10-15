"""srlab.inverse -- regularised spectral inversion (M5), lift (M6) and kappa_inv.

Charter I7: the optimizer settings are frozen in `config.InversionCfg`; every call records the
argmin residual, the optimality residual, the iteration count and `success`; `success=False` is
fatal.  M21 adds the layer split:

    solver layer     : success | retry_success | typed failure   (raise, never return garbage)
    experiment layer : correct | abstain(reason code)            (catch, log, continue)

`invert` implements the solver layer: it retries ONCE from the returned point (the frozen
deterministic warm restart) and then raises `InversionFailure` carrying its diagnostics.  The
experiment layer is `invert_or_abstain`, which converts that event into a registered abstention
with `reason code = solver_nonconvergence` so that a campaign is never aborted (aborting would
discard runs and bias the data).

Gamma rule (M19, 2026-09-28): ONE rule, scale-free, gamma_N = gamma_rel / R_hat_N with
R_hat_N = mean(ev^2) = d^-1 tr((S_r^c)^2).  Note the parentheses: the paper material wrote
d^-1 tr(S_r^c)^2, which differs by a factor 31.7 in regime A; the CODE is the calibrated
convention and M19 corrects the document.  The pre-M19 rule survives only as the frozen baseline
`config.INV_V1` used by the golden regression until the M25 window closes -- it is not a
supported experiment path.
"""
import numpy as np
from scipy.optimize import minimize

from .config import InversionCfg, INV_TABLEI_STRONG, SOLVER
from .invariants import ConvergenceError, InvariantError, abstention_reason, require
from .forward import law_zero_atom
from . import solver
from .sampling import split_atom, emp_stieltjes_plus
from .stats import wasserstein, quantiles_discrete


class InversionFailure(InvariantError):
    """Typed solver-layer failure event (M21).  Carries the diagnostics of both attempts."""

    def __init__(self, msg, info):
        super().__init__(msg)
        self.info = info


def zgrid_from_eigs(eig_pos, n_x, h_rel=0.03, x_lo_rel=0.0):
    """Fixed z-grid: x in [x_lo, x_hi], n_x points, Im z = h_rel * x_hi.
    NOTE (audit): h_rel here is the d_St bandwidth (the metric's smoothing), NOT the Im z of
    forward.law_quantiles -- the two `h`s are different objects and must not be conflated."""
    x_hi = 1.05 * np.quantile(eig_pos, 0.995)
    x = np.linspace(x_lo_rel * x_hi, x_hi, n_x)
    return x + 1j * h_rel * x_hi


def gamma_value(cfg, ev):
    """The single regularisation rule (M19), plus the frozen pre-M19 baseline for the regression."""
    if cfg.gamma_rule == "scale_free":
        return cfg.gamma_rel / max(float(np.mean(np.asarray(ev, float) ** 2)), 1e-12)
    if cfg.gamma_rule == "legacy_N":
        return cfg.gamma0 / cfg._N  # only set by Problem for the frozen baseline
    raise InvariantError(f"[I7] unknown gamma rule {cfg.gamma_rule!r}")


class Problem:
    """Empirical target + count law; evaluates the loss and its analytic gradient.

    Diagnostics that were previously implicit are counted, never hidden:
      * `n_warm_fallback` -- how many times the cheap warm start failed its charter checks and the
        full continuation solve had to be used (the archive did this SILENTLY; the arithmetic here
        is identical, the invisibility is what M-charter forbids);
      * `max_resid`       -- worst per-point fixed-point residual seen across all solves.
    """

    def __init__(self, eigs, d, N, nu, tau, zgrid=None, cfg=INV_TABLEI_STRONG, legacy_baseline=False):
        self.d, self.N, self.c = int(d), int(N), d / N
        self.cfg = cfg
        self.pos, self.p0_emp = split_atom(eigs)
        if zgrid is None:
            self.z = zgrid_from_eigs(self.pos, cfg.n_x, cfg.h_rel, cfg.x_lo_rel)
        else:
            self.z = np.asarray(zgrid, complex)
        self.m_emp = emp_stieltjes_plus(self.pos, d, self.z)
        self.nu = np.asarray(nu, float)
        self.t = np.arange(len(self.nu), dtype=float)
        self.tau = np.asarray(tau, float)
        self.p0_model = law_zero_atom(np.zeros(len(self.tau)), self.tau, float(self.nu[0]), self.c)
        R_hat = max(float(np.mean(np.asarray(eigs, float) ** 2)), 1e-12)
        if cfg.gamma_rule == "legacy_N":
            require(legacy_baseline, "I7",
                    "gamma_rule='legacy_N' is the pre-M19 frozen baseline only; "
                    "experiments must use the scale-free rule (M19)")
            self.gamma = cfg.gamma0 / N
        else:
            self.gamma = cfg.gamma_rel / R_hat
        self.R_hat = R_hat
        self._e = None
        self.n_warm_fallback = 0
        self.max_resid = 0.0
        self.n_solves = 0

    # ------------------------------------------------------------------ forward model
    def model(self, g, jac=False):
        m = None
        if self._e is not None:
            e, A, D, resid, _ = solver.solve_e(self.z, self.tau, g, self.t, self.nu, self.c,
                                               e0=self._e, maxit=SOLVER.warm_maxit)
            self.n_solves += 1
            ok = bool(np.all(np.isfinite(e)) and resid.max() < SOLVER.accept_resid
                      and e.imag.min() > 0)
            if ok:
                self.max_resid = max(self.max_resid, float(resid.max()))
                m = (np.asarray(g)[None, :] / D).sum(1)
            else:
                # VISIBLE fallback: counted, then the correctly converging path is used.
                self.n_warm_fallback += 1
        if m is None:
            m, e, A, D, info = solver.stieltjes(self.z, self.tau, g, self.t, self.nu, self.c,
                                                accept=SOLVER.accept_resid)
            self.max_resid = max(self.max_resid, info["max_resid"])
            self.n_solves += 1
        self._e = e
        mplus = m + self.p0_model / self.z
        if not jac:
            return mplus
        return mplus, solver.jacobian_m(self.tau, g, self.t, self.nu, self.c, e, A, D)

    def loss_grad(self, g):
        g = np.maximum(g, 0.0)
        mplus, J = self.model(g, jac=True)
        r = mplus - self.m_emp
        L = float(np.mean(np.abs(r) ** 2) + self.gamma * np.dot(g, self.tau ** 2))
        grad = 2.0 * np.mean((np.conj(r)[:, None] * J).real, axis=0) + self.gamma * self.tau ** 2
        return L, grad

    def dst(self, g):
        """d_St at g (the argmin residual actually reported)."""
        return float(np.sqrt(np.mean(np.abs(self.model(g) - self.m_emp) ** 2)))


# ---------------------------------------------------------------------- the solver layer
def _run_slsqp(prob, g0, maxiter, ftol):
    M = len(prob.tau)
    L0, _ = prob.loss_grad(g0)
    sc = 1.0 / max(L0, 1e-12)          # objective scaled to O(1) for SLSQP (frozen choice)
    f = lambda g: tuple(v * sc for v in prob.loss_grad(g))
    res = minimize(f, g0, jac=True, method="SLSQP",
                   bounds=[(0.0, 1.0)] * M,
                   constraints=[{"type": "eq", "fun": lambda g: g.sum() - 1.0,
                                 "jac": lambda g: np.ones_like(g)}],
                   options={"maxiter": maxiter, "ftol": ftol})
    g = np.maximum(res.x, 0.0)
    s = g.sum()
    g = g / s if s > 0 else np.full(M, 1.0 / M)  # I9: gap = 1 - raw_sum, recorded in info (SLSQP simplex projection)
    # I7 optimality residual.  The diagnostic evaluation must not perturb the warm-start state that
    # the historical call sequence (dst, then kappa_hat) starts from -- otherwise eps_model and
    # kappa_hat move in the last ulps (found in the P3 EXP-1 migration check, 2026-09-29).
    e_saved = prob._e
    _, grad = prob.loss_grad(g)
    prob._e = e_saved
    return g, dict(nit=int(res.nit), success=bool(res.success), msg=str(res.message),
                   grad_resid=float(np.max(np.abs(grad))), simplex_resid=float(abs(g.sum() - 1.0)),
                   raw_min=float(res.x.min()), raw_sum=float(s))


def invert(prob, g0=None, maxiter=None, ftol=None, restarts=None):
    """Solver layer (M21).  Returns (g, info); raises InversionFailure if no attempt succeeded.

    `init` is frozen in config: "uniform" -> g0 = 1/M on the grid.  On failure the frozen
    deterministic warm restart is taken from the returned point, ONCE (config.restarts = 1);
    if that also fails, the typed failure event is raised."""
    cfg = prob.cfg
    maxiter = cfg.maxiter if maxiter is None else maxiter
    ftol = cfg.ftol if ftol is None else ftol
    n_restart = cfg.restarts if restarts is None else restarts
    if g0 is None:
        require(cfg.init == "uniform", "I7", f"unknown init {cfg.init!r}")
        g0 = np.full(len(prob.tau), 1.0 / len(prob.tau))

    attempts = []
    g, meta = _run_slsqp(prob, g0, maxiter, ftol)
    info = dict(status="success", attempts=1, max_solver_resid=prob.max_resid,
                n_warm_fallback=prob.n_warm_fallback, gamma=prob.gamma, R_hat=prob.R_hat,
                eps_model=prob.dst(g), **meta)
    attempts.append(dict(attempt=1, **meta))
    if not meta["success"]:
        for k in range(n_restart):
            g2, meta2 = _run_slsqp(prob, g, maxiter, ftol)
            attempts.append(dict(attempt=2 + k, **meta2))
            if meta2["success"]:
                g, meta = g2, meta2
                # eps_model must describe the RETURNED g (it used to keep the failed attempt's value)
                info.update(status="retry_success", attempts=2 + k, eps_model=prob.dst(g), **meta)
                break
        else:
            raise InversionFailure(
                f"[I7] inversion failed after {1 + n_restart} attempt(s): {meta['msg']}",
                dict(attempts=attempts, eps_model=prob.dst(g), gamma=prob.gamma))
    info["attempts_detail"] = attempts
    return g, info


def invert_baseline(prob, g0=None):
    """FROZEN-BASELINE ONLY (P3 migration checks).  The historical srinv.invert contract: ONE SLSQP
    attempt, no retry, no typed failure -- the returned g is used even when success=False, exactly as
    the archived EXP-3 did (M21: two forced_uniform steps fed an unconverged g into the trajectory).
    Exists so the migration can reproduce the archive bitwise; the status is returned and callers
    record it in every row.  Never used by a path that produces paper numbers (M21 applies there)."""
    require(prob.cfg.gamma_rule == "legacy_N", "I7", "invert_baseline is for the frozen pre-M19 baseline only")
    if g0 is None:
        g0 = np.full(len(prob.tau), 1.0 / len(prob.tau))
    g, meta = _run_slsqp(prob, g0, prob.cfg.maxiter, prob.cfg.ftol)
    return g, dict(status="success" if meta["success"] else "legacy_unconverged_used",
                   eps_model=prob.dst(g), **meta)


def invert_or_abstain(prob, g0=None, reason="solver_nonconvergence"):
    """Experiment layer (M21): the typed failure becomes a registered abstention, never an abort.

    Returns (g | None, record).  `record["abstain"]` is True and carries the reason code; the
    caller routes the trajectory down the fallback path and logs the event.

    O-1(c) / M38: catches `ConvergenceError` (plus `InversionFailure`, which is not a subclass of it)
    rather than one narrow type.  `_run_slsqp` -> `Problem.model` -> `solver.stieltjes` ->
    `check_converged` raises the BASE `ConvergenceError` from deep inside the objective, so an
    inversion whose forward solve fails to converge used to escape this handler and abort the whole
    campaign.  That is the one thing the charter forbids: a campaign must end with a recorded
    abstention, never with a traceback.  The reason code is preserved when the callee supplies one."""
    try:
        g, info = invert(prob, g0=g0)
        return g, dict(abstain=False, reason=None, info=info)
    except (InversionFailure, ConvergenceError) as exc:
        code = abstention_reason(exc) if isinstance(exc, ConvergenceError) else reason
        if code == ConvergenceError.reason_code:      # keep the caller's quantity-specific default
            code = reason
        return None, dict(abstain=True, reason=code,
                          detail={"message": str(exc), **getattr(exc, "info", {})})


# ------------------------------------------------------------------------ diagnostics
def kappa_hat(prob, g, supp_tol=1e-4):
    """Linearised local condition number of the inversion, over feasible mass moves e_i - e_j
    with j in supp(g): kappa^(W1) = min over j of RMS(J_i - J_j)/|tau_i - tau_j|."""
    _, J = prob.model(g, jac=True)
    supp = np.nonzero(g > supp_tol)[0]
    require(len(supp) > 0, "I5", "kappa_hat: empty support")
    best = np.inf
    for j in supp:
        diff = J - J[:, [j]]
        num = np.sqrt(np.mean(np.abs(diff) ** 2, axis=0))
        den = np.abs(prob.tau - prob.tau[j])
        ok = den > 0
        if np.any(ok):
            best = min(best, float(np.min(num[ok] / den[ok])))
    return best


def lift_eigs(g, tau, d):
    """M6 lift: h_j = Q_Hhat((j-1/2)/d), returned ascending (matches eigvalsh order)."""
    u = (np.arange(1, d + 1) - 0.5) / d
    return np.maximum(quantiles_discrete(tau, g, u), 0.0)


def w_to(g, tau, xH, wH, p=1):
    """W_p between the discrete law (tau, g) and the reference law (xH, wH)."""
    return wasserstein(tau, g, xH, wH, p)
