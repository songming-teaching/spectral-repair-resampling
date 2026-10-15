"""srlab.laws -- count laws nu (pmf on 0..K).  Function bodies are verbatim copies of the
historical implementation (bitwise reproduction); every constructor used by experiments returns a
charter-checked law (I1: unit mass and m1(nu) = 1)."""
import numpy as np
from math import lgamma
from scipy.special import gammaln
from .invariants import check_count_law, normalise


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
    for s in np.unique(shifts):
        L = lams[shifts == s]
        kk = k - s
        valid = kk >= 0
        kv = kk[valid]
        with np.errstate(divide="ignore"):
            logL = np.log(np.maximum(L, 1e-300))[:, None]
        lp = kv[None, :] * logL - L[:, None] - gammaln(kv + 1)[None, :]
        lp[:, kv == 0] = -L[:, None]
        pmf[valid] += np.exp(lp).sum(axis=0)
    pmf /= len(lams)  # I9: definitional -- the mixture average (1/N) sum_i, not a renormalisation
    return pmf


def trim_pmf(pmf, tol=1e-15):
    last = np.nonzero(pmf > tol)[0]
    return pmf[: last[-1] + 1] if len(last) else pmf[:1]


def _raw_plugin_multi(lam):
    return poisson_mixture_pmf(lam)


def _raw_plugin_res(lam):
    k = np.floor(lam).astype(int)
    u = lam - k
    return poisson_mixture_pmf(u, shifts=k)


def _raw_realized(counts, kmax=None):
    counts = np.asarray(counts, int)
    kmax = int(counts.max()) if kmax is None else kmax
    return np.bincount(counts, minlength=kmax + 1)[: kmax + 1] / len(counts)


def _unnormalised_oracle_multi_gamma(alpha, kmax=None):
    """NegBin pmf on 0..kmax WITHOUT renormalisation (the gap is the truncated tail; I9 checks it)."""
    if kmax is None:
        kmax = _kmax(1.0 + 30.0 / np.sqrt(alpha) + 30.0 / alpha)
    k = np.arange(kmax + 1)
    return np.exp(gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1)
                  + alpha * np.log(alpha / (alpha + 1.0)) - k * np.log(alpha + 1.0))


def _raw_oracle_multi_gamma(alpha, kmax=None):
    """MixPoi(Gamma(alpha, rate alpha)) = NegBin(r=alpha, p=alpha/(alpha+1))."""
    if kmax is None:
        kmax = _kmax(1.0 + 30.0 / np.sqrt(alpha) + 30.0 / alpha)
    k = np.arange(kmax + 1)
    lp = (gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1)
          + alpha * np.log(alpha / (alpha + 1.0)) - k * np.log(alpha + 1.0))
    pmf = np.exp(lp)
    return pmf / pmf.sum()  # I9: FROZEN-BASELINE ONLY (verbatim legacy arithmetic for the migration check); guarded constructors go through normalise()


def _raw_oracle_res_gamma(alpha, kmax=None, nquad=400):
    """L(K + Poi(U)), (K,U) = (floor, frac) of lambda ~ Gamma(alpha, rate alpha)."""
    from scipy.stats import gamma as gam
    if kmax is None:
        kmax = _kmax(1.0 + 30.0 / np.sqrt(alpha) + 30.0 / alpha)
    pmf = np.zeros(kmax + 1)
    xg, wg = np.polynomial.legendre.leggauss(nquad)
    for K in range(kmax + 1):
        lo, hi = K, K + 1.0
        if K == 0:
            s = 0.5 * (xg + 1.0)
            lamv = s ** (1.0 / alpha)
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
    return pmf / pmf.sum()  # I9: FROZEN-BASELINE ONLY (verbatim legacy arithmetic for the migration check); guarded constructors go through normalise()


def _oracle_res_gamma_exact(alpha, kmax=None, nquad=400):
    """L(K + Poi(U)), (K, U) = (floor, frac) of lambda ~ Gamma(alpha, rate alpha) -- M34 fix of D23.

    pmf(n) = sum_K int_K^{K+1} f(lambda) Poi(n - K; lambda - K) dlambda.
      K = 0 : CLOSED FORM.  int_0^1 lambda^(a-1) e^{-b lambda} dlambda = Gamma(a) P(a, b) / b^a with
              a = alpha + j, b = alpha + 1 (P = regularised lower incomplete gamma).  The historical
              substitution lambda = s^(1/alpha) made this integrand non-smooth and Gauss-Legendre
              converged only algebraically (m1 - 1 = 1.9e-4 at alpha = 10).
      K >= 1: mu = lambda - K in [0, 1]; the integrand (mu + K)^(alpha-1) e^{-alpha(mu+K)} e^{-mu} mu^j
              is smooth, Gauss-Legendre (nquad points) is accurate.
    Returns (pmf WITHOUT renormalisation, pre-normalisation mass gap) -- the caller runs I9."""
    from scipy.special import gammainc
    if kmax is None:
        kmax = _kmax(1.0 + 30.0 / np.sqrt(alpha) + 30.0 / alpha)
    lognorm = alpha * np.log(alpha) - gammaln(alpha)          # Gamma(alpha, rate alpha) density constant
    pmf = np.zeros(kmax + 1)
    j = np.arange(kmax + 1, dtype=float)
    a = alpha + j
    b = alpha + 1.0
    # K = 0: pmf[j] += C / j! * Gamma(a) P(a, b) / b^a
    pmf += np.exp(lognorm - gammaln(j + 1) + gammaln(a) - a * np.log(b)) * gammainc(a, b)
    xg, wg = np.polynomial.legendre.leggauss(nquad)
    mu = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    for K in range(1, kmax + 1):
        lam = mu + K
        base = np.exp(lognorm + (alpha - 1.0) * np.log(lam) - alpha * lam - mu)      # f(lambda) e^{-mu}
        jj = np.arange(kmax + 1 - K, dtype=float)
        with np.errstate(divide="ignore"):
            lmu = np.log(mu)
        pois = np.exp(jj[:, None] * lmu[None, :] - gammaln(jj + 1)[:, None])        # mu^j / j!
        pmf[K:] += (pois * (w * base)[None, :]).sum(1)
    return pmf, 1.0 - float(pmf.sum())


# ------------------------------------------------------------------ checked public constructors
def plugin_multi(lam):
    return _checked(trim_pmf(_raw_plugin_multi(lam)), "plug-in multinomial")

def plugin_res(lam):
    return _checked(trim_pmf(_raw_plugin_res(lam)), "plug-in residual")

def realized(counts):
    return _checked(trim_pmf(_raw_realized(counts)), "realized counts")

def oracle(scheme, alpha):
    """Oracle count law (M34): residual scheme uses the exact construction; both schemes pass the I9
    pre-normalisation sentinel and then I1 at 1e-12 (was 1e-7, which let D23 through at alpha = 3 only
    by luck of the threshold and not at all at alpha = 10)."""
    if scheme == "multi":
        raw = _unnormalised_oracle_multi_gamma(alpha)
    else:
        raw, _gap = _oracle_res_gamma_exact(alpha)
    p, _gap = normalise(raw, f"oracle {scheme} Gamma({alpha})", tol=1e-12)
    return _checked(trim_pmf(p), f"oracle {scheme} Gamma({alpha})", tol=1e-12)

def oracle_legacy(scheme, alpha):
    """FROZEN-BASELINE ONLY (P3 migration check).  The historical oracle law WITHOUT the I1 guard,
    returned together with its first-moment error so the caller records it in every output row.
    Found 2026-09-29 (ledger D23): the residual oracle law's quadrature leaves m1 - 1 = +1.9e-4 at
    alpha = 10 and +2.1e-7 at alpha = 3, masked by the final pmf / pmf.sum() normalisation.  The
    guarded constructor `oracle()` refuses these laws; this one exists only so the migration can
    reproduce the archived rows -- it must never feed a result that enters the paper."""
    raw = _raw_oracle_multi_gamma(alpha) if scheme == "multi" else _raw_oracle_res_gamma(alpha)
    p = trim_pmf(raw)
    m1_err = float(p @ np.arange(len(p), dtype=float)) - 1.0
    return p, m1_err


def _checked(p, what, tol=1e-9):
    check_count_law(p, tol=tol, what=what)
    return p

def zero_atom(nu0, c, g_atom0=0.0):
    return max(g_atom0, max(0.0, 1.0 - (1.0 - nu0) / c))
