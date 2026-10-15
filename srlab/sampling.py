"""srlab.sampling -- weights, resampling and resampled spectra (historical arithmetic).

The boilerplate  S = X diag(n) X^T / N - m m^T ; eigvalsh  used to be copied in ten scripts; it
lives here once, with the I2 check (zero atom = count-rank identity) on every spectrum returned.
"""
import numpy as np
from .invariants import check_rank_identity


def gamma_weights(rng, alpha, N):
    """lambda_i = N lt_i / sum lt,  lt ~ Gamma(alpha, rate alpha)  (draw order is part of the seed contract)."""
    lt = rng.gamma(alpha, 1.0 / alpha, N)
    return N * lt / lt.sum()  # I9: definitional self-normalisation of the weights, not a repaired integral


def multinomial(rng, lam):
    N = len(lam)
    return rng.multinomial(N, lam / N)


def residual(rng, lam):
    """Integer-boundary protocol: floors kept as integers, R_N formed in integer arithmetic."""
    N = len(lam)
    k = np.floor(lam).astype(int); u = lam - k; RN = N - int(k.sum())
    return k + (rng.multinomial(RN, u / u.sum()) if RN > 0 else 0)  # I9: definitional -- remainder-draw probabilities, u.sum() = R_N exactly


def centred_S(X, n):
    """(1/N) X diag(n) X^T - m m^T with m = X n / N (historical expression)."""
    N = X.shape[1]
    mr = X @ n / N
    return (X * n[None, :]) @ X.T / N - np.outer(mr, mr)


def spectrum(X, n, check=True, clip=True):
    """Eigenvalues of the centred resampled covariance, charter-checked (I2).
    `clip` applies the historical max(ev, 0)."""
    ev = np.linalg.eigvalsh(centred_S(X, n))
    if check:
        d = X.shape[0]
        thr = 1e-9 * max(ev.max(), 1e-300)
        p0 = float(np.mean(ev <= thr))
        check_rank_identity(p0, int(np.count_nonzero(n)), d, centred=True)
    return np.maximum(ev, 0.0) if clip else ev


def split_atom(eigs, rel=1e-9):
    eigs = np.asarray(eigs, float)
    thr = rel * max(eigs.max(), 1e-300)
    zero = eigs <= thr
    return eigs[~zero], zero.mean()


def emp_stieltjes_plus(eig_pos, d, z):
    """Stieltjes transform of the nonzero part of the ESD (mass #pos/d)."""
    return (1.0 / (eig_pos[None, :] - z[:, None])).sum(1) / d
