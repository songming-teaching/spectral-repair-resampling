"""
srcore.py -- core numerics for the Spectral Repair experiments.

Forward model.  For a discrete population law G = sum_j g_j delta_{tau_j} and a
discrete count law nu = sum_k p_k delta_{t_k}, the limiting spectral law of
    S = (1/N) Sigma^{1/2} Z D Z^T Sigma^{1/2},   d/N -> c,
is mu = G boxtimes T_c(nu).  Its Stieltjes transform m(z) is given by the
separable (Paul-Silverstein / Zhang) fixed point
    A(e)  = sum_k p_k t_k / (1 + c t_k e)
    e     = sum_j g_j tau_j / (tau_j A(e) - z)
    m(z)  = sum_j g_j       / (tau_j A(e) - z).
For G = delta_1 this reduces to the companion-paper equation
    z = -1/m + int t/(1+c t m) nu(dt).
The zero atom of mu is max(G({0}), 1 - (1 - nu({0}))/c)  (G has no atom here).
"""
import numpy as np
from math import lgamma
from scipy.special import gammaln

# ----------------------------------------------------------------------------
# count laws
# ----------------------------------------------------------------------------
def _kmax(lmax):
    return int(np.ceil(lmax + 12.0 * np.sqrt(lmax + 1.0) + 25))

def poisson_mixture_pmf(lams, shifts=None, kmax=None):
    """(1/N) sum_i L(shift_i + Poi(lam_i)) as a pmf on 0..kmax."""
    lams = np.asarray(lams, float)
    shifts = np.zeros_like(lams, dtype=int) if shifts is None else np.asarray(shifts, int)
    if kmax is None:
        kmax = _kmax(float(np.max(lams + shifts)))
    k = np.arange(kmax + 1)
    pmf = np.zeros(kmax + 1)
    # group by shift for efficiency
    for s in np.unique(shifts):
        L = lams[shifts == s]
        kk = k - s
        valid = kk >= 0
        kv = kk[valid]
        with np.errstate(divide="ignore"):
            logL = np.log(np.maximum(L, 1e-300))[:, None]
        lp = kv[None, :] * logL - L[:, None] - gammaln(kv + 1)[None, :]
        lp[:, kv == 0] = -L[:, None]          # handles L = 0 exactly
        pmf[valid] += np.exp(lp).sum(axis=0)
    pmf /= len(lams)
    return pmf

def nu_plugin_multi(lam):
    return poisson_mixture_pmf(lam)

def nu_plugin_res(lam):
    k = np.floor(lam).astype(int)
    u = lam - k
    return poisson_mixture_pmf(u, shifts=k)

def nu_realized(counts, kmax=None):
    counts = np.asarray(counts, int)
    kmax = int(counts.max()) if kmax is None else kmax
    return np.bincount(counts, minlength=kmax + 1)[: kmax + 1] / len(counts)

def nu_oracle_multi_gamma(alpha, kmax=None):
    """MixPoi(Gamma(alpha, rate alpha)) = NegBin(r=alpha, p=alpha/(alpha+1))."""
    if kmax is None:
        kmax = _kmax(1.0 + 30.0 / np.sqrt(alpha) + 30.0 / alpha)
    k = np.arange(kmax + 1)
    lp = (gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1)
          + alpha * np.log(alpha / (alpha + 1.0)) - k * np.log(alpha + 1.0))
    pmf = np.exp(lp)
    return pmf / pmf.sum()

def nu_oracle_res_gamma(alpha, kmax=None, nquad=400):
    """L(K + Poi(U)), (K,U) = (floor, frac) of lambda ~ Gamma(alpha, rate alpha)."""
    from scipy.stats import gamma as gam
    if kmax is None:
        kmax = _kmax(1.0 + 30.0 / np.sqrt(alpha) + 30.0 / alpha)
    pmf = np.zeros(kmax + 1)
    xg, wg = np.polynomial.legendre.leggauss(nquad)
    for K in range(kmax + 1):
        # integrate over lambda in [K, K+1)
        lo, hi = K, K + 1.0
        if K == 0:
            # substitution to handle the lambda^{alpha-1} singularity at 0
            s = 0.5 * (xg + 1.0)                # in (0,1)
            lamv = s ** (1.0 / alpha)            # lambda = s^{1/alpha}
            # dlambda = (1/alpha) s^{1/alpha - 1} ds ; density f = C lam^{a-1} e^{-a lam}
            C = np.exp(alpha * np.log(alpha) - lgamma(alpha))
            dens_times_jac = C * np.exp(-alpha * lamv) / alpha
            wts = 0.5 * wg * dens_times_jac
        else:
            lamv = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
            wts = 0.5 * (hi - lo) * wg * gam.pdf(lamv, alpha, scale=1.0 / alpha)
        u = lamv - K
        for n in range(K, kmax + 1):
            j = n - K
            pois = np.exp(j * np.log(np.maximum(u, 1e-300)) - u - lgamma(j + 1)) if j > 0 else np.exp(-u)
            pmf[n] += np.sum(wts * pois)
    return pmf / pmf.sum()

def trim_pmf(pmf, tol=1e-15):
    last = np.nonzero(pmf > tol)[0]
    return pmf[: last[-1] + 1] if len(last) else pmf[:1]

# ----------------------------------------------------------------------------
# forward fixed point
# ----------------------------------------------------------------------------
def solve_e(z, tau, g, t, p, c, e0=None, maxit=4000, tol=1e-13):
    """Solve the separable fixed point for e(z) at an array of z (Im z > 0).
    Damped iteration followed by Newton polish.  Returns (e, A, D, resid)."""
    z = np.asarray(z, complex)
    tau = np.asarray(tau, float); g = np.asarray(g, float)
    t = np.asarray(t, float); p = np.asarray(p, float)
    a1 = float(np.dot(g, tau))
    e = (-a1 / z) if e0 is None else np.array(e0, complex)
    for it in range(maxit):
        A = (p[None, :] * t[None, :] / (1.0 + c * t[None, :] * e[:, None])).sum(1)
        D = tau[None, :] * A[:, None] - z[:, None]
        F = (g[None, :] * tau[None, :] / D).sum(1)
        # Newton step on  R(e) = e - F(e)
        Ap = -c * (p[None, :] * t[None, :] ** 2 / (1.0 + c * t[None, :] * e[:, None]) ** 2).sum(1)
        dF = -(g[None, :] * tau[None, :] ** 2 / D ** 2).sum(1) * Ap
        R = e - F
        step = R / (1.0 - dF)
        enew = e - step
        # safeguard: keep Im e > 0 (Stieltjes branch); otherwise damped fixed point
        bad = (enew.imag <= 0) | ~np.isfinite(enew)
        enew[bad] = 0.5 * e[bad] + 0.5 * F[bad]
        if np.max(np.abs(enew - e)) < tol * (1.0 + np.max(np.abs(e))):
            e = enew
            break
        e = enew
    A = (p[None, :] * t[None, :] / (1.0 + c * t[None, :] * e[:, None])).sum(1)
    D = tau[None, :] * A[:, None] - z[:, None]
    resid = np.abs(e - (g[None, :] * tau[None, :] / D).sum(1))
    return e, A, D, resid

def stieltjes(z, tau, g, t, p, c, continuation=True):
    """m(z) of G boxtimes T_c(nu).  Uses continuation in Im z for robustness."""
    z = np.asarray(z, complex)
    e = None
    if continuation:
        for s in [256.0, 64.0, 16.0, 4.0]:
            e, A, D, _ = solve_e(z.real + 1j * z.imag * s, tau, g, t, p, c, e0=e)
    e, A, D, resid = solve_e(z, tau, g, t, p, c, e0=e)
    m = (np.asarray(g)[None, :] / D).sum(1)
    return m, e, A, D, resid

def zero_atom(nu0, c, g_atom0=0.0):
    return max(g_atom0, max(0.0, 1.0 - (1.0 - nu0) / c))

def jacobian_m(tau, g, t, p, c, e, A, D):
    """dm/dg_j (n_z x M), holding the count law fixed."""
    tau = np.asarray(tau, float); g = np.asarray(g, float)
    Ap = -c * (p[None, :] * t[None, :] ** 2 / (1.0 + c * t[None, :] * e[:, None]) ** 2).sum(1)
    Fe = 1.0 + (g[None, :] * tau[None, :] ** 2 / D ** 2).sum(1) * Ap
    Q = (g[None, :] * tau[None, :] / D ** 2).sum(1) * Ap
    dedg = (tau[None, :] / D) / Fe[:, None]
    return 1.0 / D - Q[:, None] * dedg

# ----------------------------------------------------------------------------
# empirical side
# ----------------------------------------------------------------------------
def split_atom(eigs, rel=1e-9):
    eigs = np.asarray(eigs, float)
    thr = rel * max(eigs.max(), 1e-300)
    zero = eigs <= thr
    return eigs[~zero], zero.mean()

def emp_stieltjes_plus(eig_pos, d, z):
    """Stieltjes transform of the nonzero part of the ESD (unnormalized: mass = #pos/d)."""
    return (1.0 / (eig_pos[None, :] - z[:, None])).sum(1) / d

# ----------------------------------------------------------------------------
# Wasserstein distances between discrete laws on R
# ----------------------------------------------------------------------------
def wasserstein(x1, w1, x2, w2, p=1):
    x1 = np.asarray(x1, float); w1 = np.asarray(w1, float)
    x2 = np.asarray(x2, float); w2 = np.asarray(w2, float)
    o1 = np.argsort(x1); o2 = np.argsort(x2)
    x1, w1, x2, w2 = x1[o1], w1[o1] / w1.sum(), x2[o2], w2[o2] / w2.sum()
    c1 = np.cumsum(w1); c2 = np.cumsum(w2)
    grid = np.unique(np.concatenate([[0.0], c1, c2]))
    grid = grid[grid <= 1.0 + 1e-15]
    mids = 0.5 * (grid[1:] + grid[:-1])
    dq = np.diff(grid)
    q1 = x1[np.minimum(np.searchsorted(c1, mids), len(x1) - 1)]
    q2 = x2[np.minimum(np.searchsorted(c2, mids), len(x2) - 1)]
    return float(np.sum(dq * np.abs(q1 - q2) ** p) ** (1.0 / p))

def quantiles_discrete(x, w, u):
    o = np.argsort(x); x = np.asarray(x)[o]; cw = np.cumsum(np.asarray(w)[o]); cw /= cw[-1]
    return x[np.minimum(np.searchsorted(cw, u), len(x) - 1)]

# ----------------------------------------------------------------------------
# general quantile function of G boxtimes T_c(nu)  (for W1 distances)
# ----------------------------------------------------------------------------
def law_zero_atom(g, tau, p_nu0, c):
    g0 = float(g[0]) if abs(float(tau[0])) < 1e-12 else 0.0
    return max(g0, max(0.0, 1.0 - (1.0 - p_nu0) / c))

def _newton_active(z, tau, g, t, p, c, e, maxit=200, tol=1e-13):
    """Per-point Newton on e = F(e) (same update and stopping rule as solve_e, but each point
    leaves the active set once converged).  Returns (e, indices still unconverged)."""
    e = e.copy(); act = np.arange(len(z))
    for _ in range(maxit):
        ea, za = e[act], z[act]
        den = 1.0 + c * t[None, :] * ea[:, None]
        A = (p * t / den).sum(1)
        Ap = -c * (p * t ** 2 / den ** 2).sum(1)
        D = tau[None, :] * A[:, None] - za[:, None]
        F = (g * tau / D).sum(1)
        dF = -(g * tau ** 2 / D ** 2).sum(1) * Ap
        en = ea - (ea - F) / (1.0 - dF)
        bad = (en.imag <= 0) | ~np.isfinite(en)
        en[bad] = 0.5 * ea[bad] + 0.5 * F[bad]
        done = np.abs(en - ea) < tol * (1.0 + np.abs(ea))
        e[act] = en; act = act[~done]
        if act.size == 0:
            break
    return e, act

FAST_STATS = {"points": 0, "stragglers": 0, "calls": 0}      # M18 diagnostics: fallback share

def _stieltjes_fast(z, tau, g, t, p, c, n_coarse=1500, block=16384):
    """m(z) on a dense real-axis-parallel grid (M18): <= n_coarse points solved by the reference
    solver (with continuation), linear interpolation as the initial guess, active-set Newton on
    the fine grid in blocks (bounded temporaries: block x len(t) complex), reference solver again
    on any straggler.  Zero-weight atoms are dropped (exact: they contribute to neither sum)."""
    keep = g > 0
    tau, g = tau[keep], g[keep]
    stride = max(1, len(z) // n_coarse)
    ic = np.unique(np.r_[np.arange(0, len(z), stride), len(z) - 1])
    _, ec, *_ = stieltjes(z[ic], tau, g, t, p, c)
    x = z.real
    e0 = np.interp(x, x[ic], ec.real) + 1j * np.interp(x, x[ic], ec.imag)
    m = np.empty_like(e0)
    for s in range(0, len(z), block):
        sl = slice(s, s + block)
        eb, left = _newton_active(z[sl], tau, g, t, p, c, e0[sl])
        if left.size:
            _, eb[left], *_ = stieltjes(z[sl][left], tau, g, t, p, c)
        A = (p * t / (1.0 + c * t[None, :] * eb[:, None])).sum(1)
        m[sl] = (g / (tau[None, :] * A[:, None] - z[sl][:, None])).sum(1)
        FAST_STATS["stragglers"] += int(left.size)
    FAST_STATS["points"] += len(z); FAST_STATS["calls"] += 1
    return m

def _law_quantiles(solver, tau, g, t, p, c, u, n_x, eta_rel, x_hi, m1_tol):
    tau = np.asarray(tau, float); g = np.asarray(g, float)
    t = np.asarray(t, float); p = np.asarray(p, float)
    p0 = law_zero_atom(g, tau, float(p[0]), c)
    mean = float(g @ tau) * float(p @ t)                     # int x dmu  (first-moment multiplicativity)
    eta = eta_rel * mean
    if x_hi is None:
        x_hi = max(1.05 * float(np.max(tau[g > 1e-12])) * (1.0 + np.sqrt(c)) ** 2, 1e-3)
        grow = True
    else:
        grow = False
    while True:
        # grid spacing must resolve the smoothing scale (dx <= eta); M17 acceptance iteration 2
        xs = np.linspace(1e-6 * mean, x_hi, min(max(n_x, int(x_hi / eta) + 1), 400000))
        zz = xs + 1j * eta
        m = solver(zz, tau, g, t, p, c)
        f = np.maximum((m + p0 / zz).imag, 0.0) / np.pi
        if not grow or np.trapezoid(xs * f, xs) > (1 - m1_tol) * mean or x_hi > 1e4 * mean:
            break
        x_hi *= 1.5
    F = p0 + np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xs))])
    tot = F[-1]
    F = p0 + (F - p0) * (1.0 - p0) / max(tot - p0, 1e-12)
    q = np.interp(u, F, xs, left=0.0, right=xs[-1])
    q[u <= p0] = 0.0
    return q

def law_quantiles_ref(tau, g, t, p, c, u, n_x=40000, eta_rel=5e-5, x_hi=None, m1_tol=1e-3):
    """Quantile function Q(u) of mu = G boxtimes T_c(nu), u in [0,1) -- M17 reference version.

    M17 fix.  The density is read off at Im z = eta = eta_rel * a1 m1(nu): an ABSOLUTE
    smoothing, scaled by the spectrum but independent of the window.  (Pre-M17 versions used
    Im z = h_rel * x_hi together with a mass-capture window rule; when heavy count tails pushed
    the support past (1+sqrt c)^2 the smoothing grew with the window, mass leaked, and the
    window ran away to 1e5 -- e.g. m1 = 1.25 instead of 1.10 at c = 0.5.)  The window grows
    until the numerical first moment captures a1 m1(nu) = int x dmu to within m1_tol."""
    return _law_quantiles(lambda *a: stieltjes(*a)[0], tau, g, t, p, c, u, n_x, eta_rel, x_hi, m1_tol)

def law_quantiles(tau, g, t, p, c, u, n_x=40000, eta_rel=5e-5, x_hi=None, m1_tol=1e-3):
    """Same quantity, window, grid and quadrature as law_quantiles_ref; only the fixed-point
    solve on the dense grid is accelerated (M18: warm start + active-set Newton)."""
    return _law_quantiles(_stieltjes_fast, tau, g, t, p, c, u, n_x, eta_rel, x_hi, m1_tol)

def w1_from_quantiles(qa, qb, u):
    return float(np.trapezoid(np.abs(qa - qb), u))
