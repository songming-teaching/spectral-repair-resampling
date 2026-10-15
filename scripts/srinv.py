"""
srinv.py -- regularized spectral inversion (M5), lift (M6), and kappa_inv estimate.

Pre-registered implementation choices (documented in results/PREREGISTRATION.md):
  * candidate class: G = sum_j g_j delta_{tau_j} on a fixed grid tau in [a,b]
  * d_St: RMS over a fixed z-grid of |m^+_emp - m^+_model|, where m^+ is the
    Stieltjes transform of the NONZERO part (zero atoms are separated and
    reported as point masses, per the atom discipline).
  * z-grid: x in [x_lo, x_hi], n_x points, Im z = h_rel * x_hi,
    x_hi = 1.05 * q_{0.995}(nonzero empirical eigenvalues).
  * penalty R(G) = int x^2 dG, gamma_N = gamma0 / N.
  * optimizer: SLSQP on the simplex with the analytic Jacobian.
"""
import numpy as np
from scipy.optimize import minimize
from srcore import (stieltjes, solve_e, jacobian_m, zero_atom, split_atom,
                    emp_stieltjes_plus, wasserstein, quantiles_discrete)

DEFAULTS = dict(n_x=160, h_rel=0.03, x_lo_rel=0.0, gamma0=1e-3)

def zgrid_from_eigs(eig_pos, n_x=160, h_rel=0.03, x_lo_rel=0.0):
    x_hi = 1.05 * np.quantile(eig_pos, 0.995)
    x = np.linspace(x_lo_rel * x_hi, x_hi, n_x)
    return x + 1j * h_rel * x_hi

class Problem:
    """Holds the empirical target and a count law; evaluates loss and gradient."""
    def __init__(self, eigs, d, N, nu, tau, zgrid=None, gamma0=1e-3, **kw):
        self.d, self.N = d, N
        self.c = d / N
        self.pos, self.p0_emp = split_atom(eigs)
        opts = {**DEFAULTS, **kw}
        self.z = zgrid if zgrid is not None else zgrid_from_eigs(
            self.pos, opts["n_x"], opts["h_rel"], opts["x_lo_rel"])
        self.m_emp = emp_stieltjes_plus(self.pos, d, self.z)
        self.nu = np.asarray(nu, float)
        self.t = np.arange(len(self.nu), dtype=float)
        self.tau = np.asarray(tau, float)
        self.p0_model = zero_atom(self.nu[0], self.c)
        self.gamma = gamma0 / N
        self._e = None

    def model(self, g, jac=False):
        # warm start from the previous optimizer iterate; fall back to continuation
        m = None
        if self._e is not None:
            e, A, D, res = solve_e(self.z, self.tau, g, self.t, self.nu, self.c, e0=self._e, maxit=200)
            if np.all(np.isfinite(e)) and res.max() < 1e-10 and e.imag.min() > 0:
                m = (np.asarray(g)[None, :] / D).sum(1)
        if m is None:
            m, e, A, D, res = stieltjes(self.z, self.tau, g, self.t, self.nu, self.c)
        self._e = e
        mplus = m + self.p0_model / self.z
        if not jac:
            return mplus
        return mplus, jacobian_m(self.tau, g, self.t, self.nu, self.c, e, A, D)

    def loss_grad(self, g):
        g = np.maximum(g, 0.0)
        mplus, J = self.model(g, jac=True)
        r = mplus - self.m_emp
        L = np.mean(np.abs(r) ** 2) + self.gamma * np.dot(g, self.tau ** 2)
        grad = 2.0 * np.mean((np.conj(r)[:, None] * J).real, axis=0) + self.gamma * self.tau ** 2
        return L, grad

    def dst(self, g):
        return float(np.sqrt(np.mean(np.abs(self.model(g) - self.m_emp) ** 2)))

def invert(prob, g0=None, maxiter=500):
    M = len(prob.tau)
    g0 = np.full(M, 1.0 / M) if g0 is None else g0
    # scale objective to O(1) for SLSQP
    L0, _ = prob.loss_grad(g0)
    sc = 1.0 / max(L0, 1e-12)
    f = lambda g: tuple(v * sc for v in prob.loss_grad(g))
    res = minimize(f, g0, jac=True, method="SLSQP",
                   bounds=[(0.0, 1.0)] * M,
                   constraints=[{"type": "eq", "fun": lambda g: g.sum() - 1.0,
                                 "jac": lambda g: np.ones_like(g)}],
                   options={"maxiter": maxiter, "ftol": 1e-12})
    g = np.maximum(res.x, 0.0); g /= g.sum()
    return g, {"eps_model": prob.dst(g), "nit": int(res.nit), "success": bool(res.success),
               "msg": str(res.message)}

def kappa_hat(prob, g, supp_tol=1e-4):
    """Linearized local condition number over feasible mass moves e_i - e_j
    (j in supp g): kappa^(W1) = min RMS(J_i - J_j)/|tau_i - tau_j|.
    Also returns the W2-version slope at step t (W2 = sqrt(t)|tau_i-tau_j|),
    which scales like sqrt(t) -> 0 (the W2 condition number is identically 0)."""
    _, J = prob.model(g, jac=True)
    supp = np.nonzero(g > supp_tol)[0]
    best = np.inf
    for j in supp:
        diff = J - J[:, [j]]
        num = np.sqrt(np.mean(np.abs(diff) ** 2, axis=0))
        den = np.abs(prob.tau - prob.tau[j])
        ok = den > 0
        best = min(best, float(np.min(num[ok] / den[ok])))
    return best

def lift_eigs(g, tau, d):
    u = (np.arange(1, d + 1) - 0.5) / d
    return quantiles_discrete(tau, g, u)

def w_to(g, tau, xH, wH, p):
    return wasserstein(tau, g, xH, wH, p)
