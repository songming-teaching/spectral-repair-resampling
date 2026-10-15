"""srlab.supervisor -- the M8 spectral supervisor (specification v6, comparison rule).

    abstain  iff  (a) eps_model > tau                      [mismatch clause, unchanged]
             or   (b) Bhat_N > delta_fb                    [estimated correction error exceeds
                                                            the estimated fallback error]

with

    delta_fb := W1( Hhat boxtimes T_c(nu_realized), Hhat )      (Hhat plugs in for H^-)
    Bhat_N   := kappa_inv^-1 [ delta + sqrt((eps + delta)^2 + gamma R(Hhat)) + eps ]

The independent kappa_inv threshold clause of v5 is DELETED: it fired at the shock points -- exactly
where correction matters most -- and amplified the error 7-12x.  kappa_inv survives only as a factor
inside Bhat_N.

The M14/M15 split (see PREREGISTRATION M12) is kept: M14 evaluates the discretisation floor at a
fixed Hhat (over-subtracts, anti-conservative), M15 re-inverts per bootstrap sample (share ~0.93).
The paper cites M15; both are available.

Charter notes
  * every random stream is an explicit argument -- `eta_disc` REQUIRES an rng (no module-level
    generator, no hidden seeding), so a call is reproducible from its arguments alone;
  * `u` grids and the number of bootstrap draws are parameters with frozen defaults, never
    module constants that a caller can silently inherit.
"""
import numpy as np

from .config import SUPERVISOR
from .forward import law_quantiles, law_zero_atom
from .stats import wasserstein
from .invariants import require

U_Q = (np.arange(20001) + 0.5) / 20001


def disc_quantiles(x, w, u):
    """Quantile function of a discrete law (historical expression)."""
    o = np.argsort(x)
    x = np.asarray(x, float)[o]
    cw = np.cumsum(np.asarray(w, float)[o])
    require(abs(1.0 - cw[-1]) <= 1e-9, "I9", f"disc_quantiles: mass gap {1.0 - cw[-1]:+.3e}")
    cw /= cw[-1]  # I9: gap checked on the line above
    return x[np.minimum(np.searchsorted(cw, u), len(x) - 1)]


def delta_fb_W1(tau, g, counts, c, u=U_Q, nu_realized=None, return_info=False, legacy_unchecked=False):
    """delta_fb := W1(Hhat boxtimes T_c(nu_realized), Hhat).

    `nu_realized` may be supplied by the caller (the platform constructs it through
    `laws.realized`, which charter-checks it); otherwise it is built here from the counts."""
    if nu_realized is None:
        from .laws import realized
        nu_realized = realized(counts)
    tq = np.arange(len(nu_realized), dtype=float)
    q_joint, qinfo = law_quantiles(tau, g, tq, nu_realized, c, u, legacy_unchecked=legacy_unchecked)
    q_ref = disc_quantiles(tau, g, u)
    val = float(np.trapezoid(np.abs(q_joint - q_ref), u))
    if not return_info:
        return val
    si = qinfo.get("solver", {})
    return val, dict(cap=qinfo["cap"], coarse_unconverged=si.get("coarse_unconverged"),
                     stragglers=si.get("stragglers"), n_grid=qinfo["n_used"],
                     returned_unconverged=si.get("returned_unconverged_all_iterations"),
                     worst_unconverged_resid=si.get("worst_unconverged_resid"),
                     straggler_escalation=si.get("straggler_escalation"),
                     # a recorded design decision ①: the eta-refinement ladder actually walked on this call (level,
                     # eta_rel, n_used, mass_capture, sec per rung) -- so a guard-triggered refinement
                     # is traceable, not an invisible saving.
                     eta_ladder=qinfo.get("eta_ladder"),
                     # O-3/D25: delta_fb feeds the supervisor's clause (b) decision, so the backend
                     # behind it is part of the record.  Without this the EXP-3 supervisor rows in the
                     # freeze window would carry a decision with no traceable solver path.
                     path=si.get("path"), paths_seen=si.get("paths_seen"),
                     max_resid=si.get("max_resid"), ladder=si.get("ladder"))


def budget_Bhat(kappa_inv, eps_model, delta_gap, gamma, R_of_Hhat):
    """Theorem B1 normal form, with kappa_inv as an internal factor only."""
    k = max(float(kappa_inv), 1e-12)
    return (1.0 / k) * (delta_gap + np.sqrt((eps_model + delta_gap) ** 2 + gamma * R_of_Hhat)
                        + eps_model)


def budget_Bhat_M14(kappa_inv, eps_model, eta, delta_gap, gamma, R_of_Hhat):
    """B1 normal form with the discretisation floor removed BEFORE kappa^-1 amplification (M12)."""
    eps_t = max(float(eps_model) - float(eta), 0.0)
    return budget_Bhat(kappa_inv, eps_t, delta_gap, gamma, R_of_Hhat), eps_t


def eta_disc(prob, g, lam, rng, B=None, centered=True, refit=False):
    """Parametric-bootstrap null residual: median over B simulations from the FITTED model
    (Hhat, That) of d_St(mu_hat^(b), F_That(Hhat)), on the same z-grid and with the same atom rule.

    `rng` is REQUIRED (charter: no unseeded generator exists).  `refit=False` is M14 (fixed Hhat),
    `refit=True` is M15 (re-invert each sample).  Returns (median, all_values)."""
    from .inverse import Problem, invert

    require(rng is not None, "I5", "eta_disc requires an explicit rng (no hidden seeding)")
    B = SUPERVISOR.eta_B if B is None else B
    d, N = prob.d, prob.N
    u = (np.arange(1, d + 1) - 0.5) / d
    s = np.maximum(disc_quantiles(prob.tau, g, u), 0.0)
    vals = []
    for _ in range(int(B)):
        X = np.sqrt(s)[:, None] * rng.standard_normal((d, N))
        n = rng.multinomial(N, lam / N)
        if centered:
            mr = X @ n / N
            S = (X * n[None, :]) @ X.T / N - np.outer(mr, mr)
        else:
            S = (X * n[None, :]) @ X.T / N
        ev = np.maximum(np.linalg.eigvalsh(S), 0.0)
        pb = Problem(ev, d, N, prob.nu, prob.tau, zgrid=prob.z, cfg=prob.cfg,
                     legacy_baseline=(prob.cfg.gamma_rule == "legacy_N"))
        pb.gamma = prob.gamma                    # the bootstrap must use the SAME penalty
        if refit:
            gb, info = invert(pb, g0=g.copy())
            vals.append(float(info["eps_model"]))
        else:
            vals.append(float(pb.dst(g)))
    return float(np.median(vals)), [float(v) for v in vals]


def decide(eps_model, Bhat, delta_fb, tau, reasons=True):
    """The comparison rule.  Returns (abstain, reason) with the fired clause named."""
    if eps_model > tau:
        return True, ("mismatch_clause_a" if reasons else True)
    if Bhat > delta_fb:
        return True, ("budget_clause_b" if reasons else True)
    return False, None
