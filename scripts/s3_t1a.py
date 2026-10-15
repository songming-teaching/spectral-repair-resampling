"""S3: T1a complete comparison (pure synthetic, no filtering loop).

  (i)   m2 of weighted / multinomial / residual ensembles is exactly linear in r2,
        scheme excess c a1^2 (multi) vs c a1^2 E_pi[U] <= c a1^2 (residual);
  (ii)  p_s = 1 - E exp(-Lambda) is monotone in the convex order, p_s <= 1 - 1/e,
        and r2 alone does NOT order p_s (same-r2 counterexamples, order reversal);
  (iii) H = delta_1: zero atom = max(0, 1 - p_s / c).

Usage:
  python s3_t1a.py prereg <expdir>          # closed-form targets, BEFORE any simulation
  python s3_t1a.py run    <expdir> [workers]
"""
import os
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[v] = "1"
import sys, json, time
from fractions import Fraction as Fr
import numpy as np
from multiprocessing import Pool
from scipy.integrate import quad
from scipy.stats import poisson
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srcore import *

MASTER, EXPID = 20260925, 16          # EXPID: stable S3 identifier in SeedSequence
N, R = 2000, 50
CS = [0.5, 0.75]                      # H = delta_1 branch (d = cN; c = 0.5 uses the first 1000 rows)
DMAX = int(round(max(CS) * N))
TAUH, GH = np.array([0.5, 2.5]), np.array([0.7, 0.3])   # H0 branch, c = 0.75
A1, M2H = float(TAUH @ GH), float(TAUH ** 2 @ GH)
KEEP_ESD = {"sym_1", "spike_0.02", "gamma_1"}          # eigenvalues archived for the figure (reps < 5)

# ----------------------------------------------------------------------------
# weight laws rho (mean 1).  "iid": lambda_i iid then normalised to sum N (skeleton);
# "quota": exact quotas N*prob with exact rational atoms (sum lambda = N exactly,
# floors taken on the rationals -- integer-boundary protocol of M2).
# ----------------------------------------------------------------------------
def spike_atoms(q):
    """two-point law with mean 1, variance 1 (r2 = 2): mass q at b, 1-q at a."""
    k = Fr(q) ; s = Fr(1 - q, 1) / k        # (1-q)/q must be a perfect square here
    root = Fr(int(round(float(s) ** 0.5)))
    assert root * root == s
    return [1 - 1 / root, 1 + root], [1 - k, k]

LAWS = []
for al in [100.0, 30.0, 10.0, 3.0, 1.0, 0.5, 0.3]:
    LAWS.append(dict(name=f"gamma_{al:g}", kind="iid", fam="gamma", par=al))
for v in [0.25, 1.0, 4.0]:
    LAWS.append(dict(name=f"lognorm_{v:g}", kind="iid", fam="lognorm", par=v))
for s in [Fr(0), Fr(1, 4), Fr(1, 2), Fr(3, 4), Fr(1)]:
    LAWS.append(dict(name=f"sym_{float(s):g}", kind="quota", fam="sym", par=float(s),
                     atoms=[1 - s, 1 + s], probs=[Fr(1, 2), Fr(1, 2)]))
for q in [Fr(1, 10), Fr(1, 50)]:
    a, p = spike_atoms(q)
    LAWS.append(dict(name=f"spike_{float(q):g}", kind="quota", fam="spike", par=float(q), atoms=a, probs=p))
LAW = {L["name"]: L for L in LAWS}

# ---------------------------------------------------------------- closed forms
def _expect(L, f, lo=0.0, hi=np.inf):
    """E_rho f(Lambda) restricted to lo <= Lambda < hi."""
    if L["kind"] == "quota":
        return float(sum(float(p) * f(float(a)) for a, p in zip(L["atoms"], L["probs"])
                         if lo <= float(a) < hi))
    if L["fam"] == "gamma":
        from scipy.stats import gamma as gam
        pdf = lambda x: gam.pdf(x, L["par"], scale=1.0 / L["par"])
    else:
        sg2 = np.log1p(L["par"]); mu = -0.5 * sg2
        from scipy.stats import lognorm
        pdf = lambda x: lognorm.pdf(x, np.sqrt(sg2), scale=np.exp(mu))
    # integrate piecewise on unit intervals (frac() is discontinuous at integers)
    tot, K = 0.0, int(np.floor(lo))
    while K < hi and K < 5000:
        a_, b_ = max(lo, K), min(hi, K + 1)
        if b_ > a_:
            v, _ = quad(lambda x: f(x) * pdf(x), a_, b_, limit=400, points=[a_ + 1e-9] if K == 0 else None)
            tot += v
            if K > 3 and v < 1e-17 and abs(f(K + 1.0)) < 1e6:
                break
        K += 1
    return tot

def closed_form(L):
    Es = _expect(L, lambda x: np.exp(-x))
    Es_lt1 = _expect(L, lambda x: np.exp(-x), 0.0, 1.0)
    m1 = _expect(L, lambda x: x)
    r2 = _expect(L, lambda x: x * x)
    EU = _expect(L, lambda x: x - np.floor(x))
    ps, ps_res = 1 - Es, 1 - Es_lt1
    out = dict(name=L["name"], fam=L["fam"], par=L["par"], m1=m1, r2=r2, E_pi_U=EU,
               p_s=ps, p_s_res=ps_res, ps_lower_env=(1 - np.exp(-r2)) / r2)
    for c in CS:
        out[f"atom_multi_c{c}"] = max(0.0, 1 - ps / c)
        out[f"atom_res_c{c}"] = max(0.0, 1 - ps_res / c)
        out[f"m2_d1_weighted_c{c}"] = 1 + c * r2
        out[f"m2_d1_multi_c{c}"] = 1 + c * (1 + r2)
        out[f"m2_d1_res_c{c}"] = 1 + c * (r2 + EU)
    c = 0.75
    out["m2_H0_weighted"] = M2H + c * A1 ** 2 * r2
    out["m2_H0_multi"] = M2H + c * A1 ** 2 * (1 + r2)
    out["m2_H0_res"] = M2H + c * A1 ** 2 * (r2 + EU)
    return out

def mixpoi_pmf(L, kmax=400):
    """limit count law MixPoi(rho) as a pmf."""
    if L["fam"] == "gamma":
        return trim_pmf(nu_oracle_multi_gamma(L["par"]))
    k = np.arange(kmax + 1)
    if L["kind"] == "quota":
        pmf = sum(float(p) * poisson.pmf(k, float(a)) for a, p in zip(L["atoms"], L["probs"]))
    else:
        sg2 = np.log1p(L["par"]); mu = -0.5 * sg2
        xg, wg = np.polynomial.hermite_e.hermegauss(160)
        lam = np.exp(mu + np.sqrt(sg2) * xg); wg = wg / wg.sum()
        pmf = (wg[:, None] * poisson.pmf(k[None, :], lam[:, None])).sum(0)
    return trim_pmf(pmf / pmf.sum())

def limit_quantiles_d1(nu, c, u, eta=2e-4, n_x=40000):
    """Quantiles of delta_1 boxtimes T_c(nu).  Fixed absolute Im-part eta (NOT proportional to the
    window, cf. srcore.law_quantiles, whose window-proportional smoothing loses mass when heavy count
    tails push the support past (1+sqrt c)^2 -- pre-simulation bug, M16); the window grows until the
    first moment (exactly 1 = a1 m1(nu)) is captured to 1e-3."""
    t = np.arange(len(nu), dtype=float); one = np.array([1.0])
    p0 = law_zero_atom(one, one, float(nu[0]), c)
    x_hi = 4.0
    while True:
        xs = np.linspace(1e-6, x_hi, n_x); zz = xs + 1j * eta
        m, *_ = stieltjes(zz, one, one, t, nu, c)
        f = np.maximum((m + p0 / zz).imag, 0.0) / np.pi
        if np.trapezoid(xs * f, xs) > 1 - 1e-3 or x_hi > 400:
            break
        x_hi *= 1.5
    F = p0 + np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xs))])
    F = p0 + (F - p0) * (1.0 - p0) / max(F[-1] - p0, 1e-12)
    q = np.interp(u, F, xs, left=0.0, right=xs[-1]); q[u <= p0] = 0.0
    return q

def prereg(E):
    out = {"N": N, "R": R, "CS": CS, "H0": dict(atoms=TAUH.tolist(), weights=GH.tolist(), a1=A1, m2=M2H),
           "ps_upper": 1 - np.exp(-1.0), "laws": [closed_form(L) for L in LAWS]}
    # --- limit-law W1 distances inside the same-r2 group (H = delta_1): not only the atom differs
    grp = ["sym_1", "gamma_1", "lognorm_1", "spike_0.1", "spike_0.02"]
    u = (np.arange(20000) + 0.5) / 20000
    Q = {}
    for c in CS:
        for g in grp:
            Q[(c, g)] = limit_quantiles_d1(mixpoi_pmf(LAW[g]), c, u)
    W = {}
    for c in CS:
        for i, g in enumerate(grp):
            for h in grp[i + 1:]:
                W[f"c{c}|{g}|{h}"] = w1_from_quantiles(Q[(c, g)], Q[(c, h)], u)
            W[f"c{c}|{g}|m2_numeric"] = float(np.mean(Q[(c, g)] ** 2))   # sanity vs 1 + c(1 + r2)
    out["same_r2_group"] = grp
    out["limit_W1_same_r2"] = W
    # --- numerical check of the lower envelope (1-e^{-r2})/r2 over random 3-point laws
    rng = np.random.default_rng([MASTER, EXPID, 999])
    env = {}
    for r2 in [1.25, 1.5, 2.0, 3.0, 4.0]:
        best, nok = np.inf, 0
        for _ in range(200000):
            x = np.sort(rng.uniform(0, 4 * r2, 3)); x[0] *= rng.uniform() < 0.5
            A = np.vstack([np.ones(3), x, x * x]); b = np.array([1.0, 1.0, r2])
            try:
                p = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            if np.all(p >= -1e-12):
                nok += 1; best = min(best, 1 - float(p @ np.exp(-x)))
        env[str(r2)] = dict(closed=(1 - np.exp(-r2)) / r2, random3pt_min=best, n_feasible=nok)
    out["lower_envelope_check"] = env
    os.makedirs(os.path.join(E, "data", "s3"), exist_ok=True)
    json.dump(out, open(os.path.join(E, "data", "s3", "s3_prereg_targets.json"), "w"), indent=1)
    print(f"{'law':>12} {'r2':>7} {'p_s':>7} {'p_s_res':>7} {'E_U':>6} {'atom.5':>7} {'atom.75':>7}")
    for r in out["laws"]:
        print(f"{r['name']:>12} {r['r2']:7.4f} {r['p_s']:7.4f} {r['p_s_res']:7.4f} {r['E_pi_U']:6.4f} "
              f"{r['atom_multi_c0.5']:7.4f} {r['atom_multi_c0.75']:7.4f}   m1={r['m1']:.6f}")
    for k, v in W.items(): print(k, round(v, 5))
    for k, v in env.items(): print("envelope r2=", k, v)

# ---------------------------------------------------------------- simulation
def draw_lambda(L, rng):
    if L["kind"] == "iid":
        if L["fam"] == "gamma":
            lt = rng.gamma(L["par"], 1.0 / L["par"], N)
        else:
            sg2 = np.log1p(L["par"]); lt = rng.lognormal(-0.5 * sg2, np.sqrt(sg2), N)
        lam = N * lt / lt.sum()
        k = np.floor(lam).astype(np.int64)
        return lam, k, lam - k
    lam, k, u = [], [], []
    for a, p in zip(L["atoms"], L["probs"]):
        m = p * N; assert m.denominator == 1
        fl = a.numerator // a.denominator                 # exact floor on the rational
        lam += [float(a)] * int(m); k += [fl] * int(m); u += [float(a - fl)] * int(m)
    lam, k, u = np.array(lam), np.array(k, np.int64), np.array(u)
    perm = rng.permutation(N)
    assert int(sum(float(a) * float(p) * N for a, p in zip(L["atoms"], L["probs"])) + 0.5) == N
    return lam[perm], k[perm], u[perm]

def centred_S(X, n):
    act = n > 0
    Xa, na = X[:, act], n[act].astype(float)
    mr = Xa @ na / N
    return (Xa * na[None, :]) @ Xa.T / N - np.outer(mr, mr), mr

def m2_pair(S, mr, d):
    """(centred, uncentred) m2 from S_c and the weighted mean: A = S_c + mr mr^T."""
    t2 = float(np.sum(S * S)); q = float(mr @ S @ mr); mm = float(mr @ mr)
    return t2 / d, (t2 + 2 * q + mm * mm) / d

def task(args):
    li, rep = args
    L = LAWS[li]
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, EXPID, li, rep]))
    lam, k, u = draw_lambda(L, rng)
    RN = N - int(k.sum())
    Z = rng.standard_normal((DMAX, N))
    counts = {"multi": rng.multinomial(N, lam / N),
              "res": k + (rng.multinomial(RN, u / u.sum()) if RN > 0 else 0)}
    r2N = float(np.mean(lam ** 2)); EU_N = RN / N
    row = dict(law=L["name"], fam=L["fam"], par=L["par"], rep=rep, r2N=r2N, EU_N=EU_N, RN=RN,
               sum_u_sq=float(np.sum(u * u)))
    keep = {}
    for sc, n in counts.items():
        row[f"ps_{sc}"] = float(np.mean(n >= 1))
        row[f"b2_{sc}"] = float(np.mean(n.astype(float) ** 2))
        for c in CS:
            d = int(round(c * N)); X = Z[:d]
            S, mr = centred_S(X, n)
            ev = np.linalg.eigvalsh(S)
            _, p0 = split_atom(ev)
            row[f"atom_d1_{sc}_c{c}"] = float(p0)
            row[f"m2_d1_{sc}_c{c}"] = float(np.sum(ev ** 2) / d)
            row[f"m2u_d1_{sc}_c{c}"] = m2_pair(S, mr, d)[1]
            if L["name"] in KEEP_ESD and rep < 5 and c == 0.75:
                keep[f"{L['name']}_r{rep}_{sc}_c{c}"] = ev
    # H0 branch, c = 0.75: weighted ensemble vs the two resampling schemes (paired)
    d = int(round(0.75 * N)); d0 = int(round(0.7 * d))
    sig = np.where(np.arange(d) < d0, 0.5, 2.5)
    X = np.sqrt(sig)[:, None] * Z[:d]
    for sc, n in [("weighted", lam)] + list(counts.items()):
        S, mr = centred_S(X, n) if sc != "weighted" else centred_S(X, lam)
        row[f"m2_H0_{sc}"], row[f"m2u_H0_{sc}"] = m2_pair(S, mr, d)
    # finite-N conditional targets given lambda (uncentred; centring is O(1/d))
    HN_a1 = float(sig.mean()); HN_m2 = float((sig ** 2).mean()); cN = d / N
    Eb2 = {"weighted": r2N, "multi": 1 + r2N * (1 - 1 / N),
           "res": r2N + (EU_N - row["sum_u_sq"] / (N * RN) if RN > 0 else 0.0)}
    for sc, b in Eb2.items():
        row[f"Eb2_{sc}"] = b
        row[f"m2_H0_{sc}_condth"] = HN_m2 + b * (cN * HN_a1 ** 2 + HN_m2 / N)
    return row, keep

def run(E, W):
    out = os.path.join(E, "data", "s3"); os.makedirs(out, exist_ok=True)
    assert os.path.exists(os.path.join(out, "s3_prereg_targets.json")), "run prereg first"
    tasks = [(li, r) for li in range(len(LAWS)) for r in range(R)]
    t0, rows, arrays = time.time(), [], {}
    with Pool(W) as pool:
        for i, (row, keep) in enumerate(pool.imap_unordered(task, tasks)):
            rows.append(row); arrays.update(keep)
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(tasks)} done, {time.time()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["law"], r["rep"]))
    json.dump(rows, open(os.path.join(out, "s3_rows.json"), "w"))
    np.savez_compressed(os.path.join(out, "s3_esd.npz"), **arrays)
    print("S3 finished in", round(time.time() - t0), "s")

if __name__ == "__main__":
    mode, E = sys.argv[1], sys.argv[2]
    if mode == "prereg":
        prereg(E)
    else:
        run(E, int(sys.argv[3]) if len(sys.argv) > 3 else 30)
