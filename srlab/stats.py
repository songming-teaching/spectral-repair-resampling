"""srlab.stats -- distances and statistics (historical implementations, verbatim)."""
import numpy as np

from .invariants import normalise, require


def wasserstein(x1, w1, x2, w2, p=1):
    x1 = np.asarray(x1, float); w1 = np.asarray(w1, float)
    x2 = np.asarray(x2, float); w2 = np.asarray(w2, float)
    o1 = np.argsort(x1); o2 = np.argsort(x2)
    x1, x2 = x1[o1], x2[o2]
    w1 = normalise(w1[o1], "wasserstein w1", tol=1e-9)[0]      # I9 sentinel; same arithmetic
    w2 = normalise(w2[o2], "wasserstein w2", tol=1e-9)[0]
    c1 = np.cumsum(w1); c2 = np.cumsum(w2)
    grid = np.unique(np.concatenate([[0.0], c1, c2]))
    grid = grid[grid <= 1.0 + 1e-15]
    mids = 0.5 * (grid[1:] + grid[:-1])
    dq = np.diff(grid)
    q1 = x1[np.minimum(np.searchsorted(c1, mids), len(x1) - 1)]
    q2 = x2[np.minimum(np.searchsorted(c2, mids), len(x2) - 1)]
    return float(np.sum(dq * np.abs(q1 - q2) ** p) ** (1.0 / p))


def quantiles_discrete(x, w, u):
    o = np.argsort(x); x = np.asarray(x)[o]; cw = np.cumsum(np.asarray(w)[o])
    require(abs(1.0 - cw[-1]) <= 1e-9, "I9", f"quantiles_discrete: mass gap {1.0 - cw[-1]:+.3e}")
    cw /= cw[-1]  # I9: gap checked on the line above
    return x[np.minimum(np.searchsorted(cw, u), len(x) - 1)]


def ess_geyer(x):
    """Initial-positive-sequence ESS (Geyer), historical tableI.ess_geyer."""
    x = x - x.mean(); n = len(x)
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]; ac /= ac[0]
    tau = 1.0; t = 1
    while t + 1 < n:
        pair = ac[t] + ac[t + 1]
        if pair < 0:
            break
        tau += 2 * pair; t += 2
    return n / tau


def sidak_maxabsz(z):
    """Global test of H0: all z ~ N(0,1) independent, via max|z| with the Sidak correction.
    p = 1 - (1 - P(|Z| > max|z|))^n  (historical analyze_g21_recheck.sidak)."""
    from scipy.stats import norm
    zz = np.abs(np.asarray(z, float)); n = len(zz)
    return float(1.0 - (2 * norm.cdf(zz.max()) - 1) ** n)
