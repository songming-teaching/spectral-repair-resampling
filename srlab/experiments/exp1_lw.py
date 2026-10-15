r"""EXP-1 / LW: the Ledoit-Wolf external baseline as EXP-1's fifth inverter (M43).

Separate driver, so that the ledger-pinned results/window/exp1_m19/exp1_rows.json is never rewritten.
The draw below is a VERBATIM replay of exp1.task (same stream, same draw order: weights, cloud,
multinomial, residual), so `ev` is the very array the four structural inverters saw.  Pairing is
not assumed, it is ASSERTED: for every (N, alpha, rep, scheme) the replayed 'none' W1/W2 must equal
the archived 'none' row bitwise; any mismatch aborts the run.

Output rows carry inverter="lw" in EXP-1's schema (W1, W2 against H_N), plus n_lw, the branch taken
(p<=n / p>n) and the zero-atom mass LW assigns.

Usage:  python -m srlab.experiments.exp1_lw --ref results/window/exp1_m19/exp1_rows.json \
            --out results/window/exp1_lw [--workers 8]
"""
import argparse
import json
import os
import sys
import time

from ..threads import pin as _pin
_pin("default")                     # D25/M37: forced before numpy loads

import numpy as np                                                     # noqa: E402

from .. import config, sampling                                        # noqa: E402
from ..stats import wasserstein                                        # noqa: E402
from ..baselines.ledoit_wolf import analytical_shrinkage, n_lw         # noqa: E402

C = config.EXP1["c"]
H_ATOMS = config.H0_ATOMS


def task(args):
    N, al, rep = args
    rng = config.rng("exp1", N, int(al * 10), rep)                     # == exp1.task, line for line
    d = int(round(C * N)); d0 = int(round(0.7 * d))
    sig = np.where(np.arange(d) < d0, 0.5, 2.5)
    HN = (H_ATOMS, np.array([d0 / d, 1 - d0 / d]))
    lam = sampling.gamma_weights(rng, al, N)
    X = np.sqrt(sig)[:, None] * rng.standard_normal((d, N))
    counts = {"multi": sampling.multinomial(rng, lam), "res": sampling.residual(rng, lam)}
    rows = []
    for scheme in ("multi", "res"):
        n = counts[scheme]
        ev = sampling.spectrum(X, n, check=True, clip=False)
        evc = np.maximum(ev, 0.0)
        wts = np.full(len(ev), 1.0 / len(ev))
        none = dict(W1=wasserstein(evc, wts, *HN, 1), W2=wasserstein(evc, wts, *HN, 2))
        nobs = n_lw(n)
        t0 = time.time()
        dt = analytical_shrinkage(evc, nobs)
        sec = time.time() - t0
        rows.append(dict(N=N, d=d, alpha=al, rep=rep, scheme=scheme, inverter="lw",
                         W1=wasserstein(dt, wts, *HN, 1), W2=wasserstein(dt, wts, *HN, 2),
                         n_lw=nobs, branch="p<=n" if d <= nobs else "p>n",
                         lw_null_mass=max(0.0, 1.0 - nobs / d), lw_null_value=float(dt[0]) if d > nobs else None,
                         lw_min=float(dt.min()), lw_max=float(dt.max()), sec=sec,
                         _none_replay=none))
    return rows


def _keyed(pair):
    label, args = pair
    return label, task(args)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="archived EXP-1 rows (window m19) for the pairing assertion")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)

    with open(a.ref, encoding="utf-8") as fh:
        ref = {(r["N"], r["alpha"], r["rep"], r["scheme"]): r
               for r in json.load(fh) if r["inverter"] == "none"}
    tasks = [(N, al, r) for N in config.EXP1["Ns"] for al in config.EXP1["alphas"]
             for r in range(config.EXP1["R"])]
    os.makedirs(a.out, exist_ok=True)

    from multiprocessing import Pool
    t0 = time.time(); allrows = []; mism = []
    with Pool(config.workers(a.workers)) as pool:
        for label, rows in pool.imap_unordered(_keyed, [(f"N{t[0]}|a{t[1]}|rep{t[2]}", t) for t in tasks]):
            for r in rows:
                arch = ref[(r["N"], r["alpha"], r["rep"], r["scheme"])]
                rp = r.pop("_none_replay")
                if rp["W1"] != arch["W1"] or rp["W2"] != arch["W2"]:
                    mism.append(dict(key=label, scheme=r["scheme"], replay=rp,
                                     archived=dict(W1=arch["W1"], W2=arch["W2"])))
                allrows.append(r)
            if len(allrows) % 100 == 0:
                print(f"{len(allrows)}/{2 * len(tasks)} rows {time.time() - t0:.0f}s", flush=True)
    if mism:
        with open(os.path.join(a.out, "PAIRING_MISMATCH.json"), "w", encoding="utf-8") as fh:
            json.dump(mism, fh, indent=1)
        print(f"PAIRING FAILED: {len(mism)} 'none' rows differ from {a.ref}; nothing written", flush=True)
        return 3
    allrows.sort(key=lambda r: (r["N"], r["alpha"], r["rep"], r["scheme"]))
    with open(os.path.join(a.out, "exp1_lw_rows.json"), "w", encoding="utf-8") as fh:
        json.dump(allrows, fh)
    print(f"EXP-1/LW {len(tasks)} tasks, {len(allrows)} rows, pairing {len(allrows)}/{len(allrows)} bitwise, "
          f"{time.time() - t0:.0f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
