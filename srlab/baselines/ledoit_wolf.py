r"""srlab.baselines.ledoit_wolf -- the external, MODEL-FREE spectral baseline (M43).

Ledoit & Wolf, "Analytical nonlinear shrinkage of large-dimensional covariance matrices",
Ann. Statist. 48(5), 2020 -- the closed-form estimator (Epanechnikov kernel, locally adaptive
bandwidth h_j = n^{-1/3} lambda_j), transcribed from the authors' published reference code
(analytical_shrinkage.m).  One eigendecomposition (done by the caller) + O(m^2) arithmetic, no
iteration, no solver.  (EXPERIMENT_PLAN_SUPPLEMENT §2 said "2012"; the 2012 estimator is the
numerical QuEST inversion, not a closed form -- see the M43 execution record.)

What LW is told, and what it is NOT told (M43 positioning):
  * it receives the SAME eigenvalues as the four structural inverters (paired comparison);
  * it assumes those eigenvalues come from a sample covariance of n i.i.d. observations, i.e. a
    Marchenko-Pastur distortion of the population spectrum.  It is NOT told that the distortion is
    T_c(nu) of a resampling count law -- that is the point of the comparison.

The choice of n (registered in M43, not tuned): a resampled cloud contains duplicated particles,
and LW's n is the number of INDEPENDENT observations.  With n = N the rank-deficient spectrum has
exact zeros and the p <= n branch divides by them.  We therefore take

    n_lw = #{i : n_i > 0} - 1        (distinct particles, minus one for the centring)

which equals the rank of the centred resampled covariance; when n_lw < p the estimator takes LW's
own p > n branch (null eigenvalues receive the common value dtilde0).
"""
import numpy as np

from ..invariants import require


def analytical_shrinkage(lam, n):
    """Shrunk eigenvalues for sample eigenvalues `lam` (any order, length p) and `n` observations.

    Returns the p shrunk eigenvalues in ASCENDING order (the order of LW's reference code).
    The p - n smallest sample eigenvalues are treated as the null space when n < p, exactly as the
    reference code does (lambda(max(1, p-n+1):p))."""
    lam = np.sort(np.asarray(lam, float))
    p = lam.size
    n = int(n)
    require(n >= 2, "LW", f"ledoit_wolf: n = {n} observations is not a sample")
    m = min(p, n)
    lp = lam[p - m:]                                          # the m retained (positive) eigenvalues
    require(lp[0] > 0.0, "LW", f"ledoit_wolf: retained eigenvalue {lp[0]:.3e} <= 0 (n_lw too large?)")

    h = n ** (-1.0 / 3.0)
    H = h * lp[None, :]                                       # H[i, j] = h * lambda_j
    x = (lp[:, None] - lp[None, :]) / H                       # x[i, j] = (lambda_i - lambda_j) / H[i, j]
    s5 = np.sqrt(5.0)
    ftilde = (3.0 / 4.0 / s5) * np.mean(np.maximum(1.0 - x ** 2 / 5.0, 0.0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hftemp = (-3.0 / 10.0 / np.pi) * x + (3.0 / 4.0 / s5 / np.pi) * (1.0 - x ** 2 / 5.0) * \
            np.log(np.abs((s5 - x) / (s5 + x)))
    edge = np.isclose(np.abs(x), s5, rtol=0.0, atol=1e-12)
    Hftemp[edge] = (-3.0 / 10.0 / np.pi) * x[edge]
    Hftilde = np.mean(Hftemp / H, axis=1)

    if p <= n:
        y = p / n
        dtilde = lp / ((np.pi * y * lp * ftilde) ** 2 + (1.0 - y - np.pi * y * lp * Hftilde) ** 2)
    else:
        Hftilde0 = (1.0 / np.pi) * (3.0 / 10.0 / h ** 2 + 3.0 / 4.0 / s5 / h * (1.0 - 1.0 / 5.0 / h ** 2)
                                    * np.log((1.0 + s5 * h) / (1.0 - s5 * h))) * np.mean(1.0 / lp)
        dtilde0 = 1.0 / (np.pi * (p - n) / n * Hftilde0)
        dtilde1 = lp / (np.pi ** 2 * lp ** 2 * (ftilde ** 2 + Hftilde ** 2))
        dtilde = np.concatenate([np.full(p - n, dtilde0), dtilde1])
    require(bool(np.all(np.isfinite(dtilde))), "LW", "ledoit_wolf: non-finite shrunk eigenvalue")
    return dtilde


def n_lw(counts):
    """The registered observation count: distinct resampled particles minus one (centring)."""
    return int(np.count_nonzero(counts)) - 1
