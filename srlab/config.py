"""srlab.config -- the single source of every constant, seed stream and numerical setting.

Nothing numerical is defined anywhere else.  Values are frozen copies of what produced the
numbers in the paper (see archive/code_2026-09-27); deviations are recorded in ledger/.
"""
from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------------------------------------
# Seeds.  Every random stream is SeedSequence([MASTER, tag, *keys]).  Tags are the historical
# ones (kept for bitwise reproduction); `note` records known legacy quirks.
# ---------------------------------------------------------------------------------------------
MASTER = 20260925

@dataclass(frozen=True)
class Stream:
    tag: int
    keys: str          # meaning of the extra keys
    note: str = ""

STREAMS = {
    "exp1":            Stream(1,  "N, int(alpha*10), rep"),
    "exp2a":           Stream(2,  "rep, group index"),
    "exp2b":           Stream(3,  "int(c*100), int(alpha*10), rep"),
    "exp3_main":       Stream(4,  "rep, k"),
    "exp3_calib":      Stream(5,  "rep, k"),
    "tableI_setup":    Stream(6,  "TABLEI_SETUP_KEY[regime]",
                              "M33 (2026-09-29): every 2x2 cell its own cloud. Keys A_strong 0, G11 1, G12 2, "
                              "G21 3 -- A_strong and G11 keep their published keys (bitwise), only the formerly "
                              "shared pair G12/G21 is re-keyed. B_weak keeps key 1 (different d and c from G11: "
                              "the same key yields a different object, no pairing issue). Legacy: "
                              "'0 for A_strong else 1'."),
    "tableI_mh":       Stream(7,  "preconditioner index [, 1000+seedtag]",
                              "legacy: identical across regimes; burn-in diagnostics reuse it on purpose "
                              "(same chains as Table I)"),
    "s7_cell":         Stream(9,  "int(c*100), int(alpha*10), rep"),
    "s7_eta":          Stream(10, "int(c*100), int(alpha*10), rep, int(h*1000)"),
    "exp3_eta":        Stream(11, "rep, k"),
    "s3":              Stream(16, "law index, rep"),
    "s3_envelope":     Stream(16, "999", "passed as a plain list to default_rng (not SeedSequence)"),
    "m17_verify":      Stream(17, ""),
    "dfbN":            Stream(18, "N, int(alpha*10), rep", "M24 execution 2026-10-05: delta_fb vs N at fixed c"),
}

TABLEI_SETUP_KEY = {"A_strong": 0, "G11": 1, "G12": 2, "G21": 3, "B_weak": 1}     # M33
TABLEI_SETUP_KEY_LEGACY = {"A_strong": 0, "G11": 1, "G12": 1, "G21": 1, "B_weak": 1}


def rng(stream, *keys):
    """The only way to obtain randomness in srlab.  No unseeded generator exists."""
    s = STREAMS[stream]
    if stream == "s3_envelope":
        return np.random.default_rng([MASTER, s.tag, *keys])
    return np.random.default_rng(np.random.SeedSequence([MASTER, s.tag, *keys]))

# legacy stand-alone seeds (kept verbatim)
SEED_TEST_SOLVER = 20260925
SEED_PREFLIGHT = 7
SEED_COMPLEXITY = (0, 1)

# ---------------------------------------------------------------------------------------------
# Numerical settings
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SolverCfg:
    tol: float = 1e-13                  # fixed-point step tolerance (same as the historical solver)
    maxit: int = 4000                   # per continuation level
    continuation: tuple = (256.0, 64.0, 16.0, 4.0)
    warm_maxit: int = 200               # warm-started re-solve inside the inversion loop
    accept_resid: float = 1e-10         # per-point |e - F(e)| required on EVERY returned point (I5)

@dataclass(frozen=True)
class QuantileCfg:
    eta_rel: float = 5e-5               # Im z = eta_rel * a1 m1(nu)   (M17 iteration 3)
    n_x_min: int = 40000
    n_x_cap: int = 400000               # historical cap; see cap_policy below
    m1_tol: float = 1e-3                # window grows until the first moment is captured (I3)
    x_hi_max_rel: float = 1e4
    n_coarse: int = 1500
    block: int = 16384
    # I6 policy.  "report" records the achieved dx/eta ratio and the cap shortfall in the returned
    # info WITHOUT changing any number (the historical behaviour, so that bitwise reproduction of
    # M17/M18 results holds); "raise" makes a cap shortfall fatal.  The 2026-09-28 probe found that
    # the cap is genuinely hit in the S7 regime (H = delta_1, heavy tails): n_needed/n_cap =
    # 1.16 / 1.39 / 2.68 at c = 0.5 / 0.75 / 1.25.  Pending the M26 recorded decision, the default stays
    # "report" so that no existing number moves; switch to "raise" once the cap or the grid is fixed.
    cap_policy: str = "report"

@dataclass(frozen=True)
class InversionCfg:
    n_x: int = 160
    h_rel: float = 0.03                 # d_St bandwidth: Im z = h_rel * x_hi (metric definition, not smoothing of a density)
    x_lo_rel: float = 0.0
    method: str = "SLSQP"
    maxiter: int = 500
    ftol: float = 1e-12
    init: str = "uniform"               # g0 = 1/M on the grid
    objective_scaling: str = "L0"       # objective divided by its value at g0
    restarts: int = 1                   # deterministic warm restart from the returned point if success=False
    gamma_rule: str = "legacy_N"        # "legacy_N": gamma0/N ; "scale_free": gamma_rel / mean(ev^2)
    gamma0: float = 1e-3
    gamma_rel: float = 1e-8

@dataclass(frozen=True)
class SupervisorCfg:
    eta_B: int = 3                      # bootstrap draws for eta_disc (M12 documented B = 3)
    u_q_n: int = 20001                  # W1 quadrature nodes for delta_fb (historical)
    supp_tol: float = 1e-4              # kappa_hat support threshold


# --- inversion configurations ---------------------------------------------------------------
# M19 (2026-09-28): ONE gamma rule, scale-free, for every experiment.
INV_SCALEFREE = InversionCfg(gamma_rule="scale_free")
INV_TABLEI_STRONG = InversionCfg(gamma_rule="scale_free", n_x=400, h_rel=0.01)   # kappa(Sigma) > 10
INV_TABLEI_WEAK = InversionCfg(gamma_rule="scale_free", n_x=160, h_rel=0.03)     # kappa(Sigma) <= 10

# Pre-M19 rule, FROZEN BASELINE ONLY: reproducible reference for the golden regression while the
# M25 freeze window is open.  Problem() refuses this rule unless legacy_baseline=True, so no
# experiment path can pick it up by accident.
INV_V1_BASELINE = InversionCfg(gamma_rule="legacy_N")

SOLVER = SolverCfg()
QUANT = QuantileCfg()
SUPERVISOR = SupervisorCfg()

# ---------------------------------------------------------------------------------------------
# Shared model constants
# ---------------------------------------------------------------------------------------------
H0_ATOMS = np.array([0.5, 2.5])
H0_WEIGHTS = np.array([0.7, 0.3])
GRID_005 = np.round(np.arange(0.2, 2.6001, 0.05), 10)       # EXP-1/2, Table I (kappa <= 10)
GRID_0025 = np.round(np.arange(0.2, 2.6001, 0.025), 10)     # EXP-3, S7

# ---------------------------------------------------------------------------------------------
# Experiment definitions (frozen copies of archive/code_2026-09-27)
# ---------------------------------------------------------------------------------------------
EXP1 = dict(c=0.75, Ns=(500, 1000, 2000), alphas=(10.0, 1.0, 0.3), R=50, grid="GRID_005", inv="INV_SCALEFREE")
EXP2 = dict(a=dict(N=2000, d=1500, d0=1050, R=50,
                   groups=("base", "endo10", "endo750", "corr0.5", "corr0.9")),
            b=dict(N=1000, cs=(0.5, 0.75, 1.0, 1.25), alphas=(10.0, 3.0, 1.0, 0.5, 0.3), R=20),
            grid="GRID_005", inv="INV_SCALEFREE")
EXP3 = dict(d=750, N=1000, a=0.8, q=0.2, K=32, shocks=(8, 16, 24), alpha=3.0, alpha_shock=0.3, R=20, R_calib=10,
            forced_end=tuple(range(27, 33)), forced_uniform=(5, 10, 15, 20, 25, 30),
            methods=("none", "cond", "oracle", "sup", "forced_end", "forced_uniform"),
            tau_rule="2 x median eps_model over calibration", eta_B=3, grid="GRID_0025", inv="INV_SCALEFREE")
S7 = dict(cs=(0.15, 0.30, 0.50, 0.75, 1.00, 1.25), alphas=(10.0, 3.0, 1.0, 0.3), D=300, R=3,
          hrels=(0.03, 0.01, 0.003), deployed_h=0.03, NX=160, eta_B=3, inv="INV_SCALEFREE")
S3 = dict(N=2000, R=50, cs=(0.5, 0.75))
# M24 execution (EXPERIMENT_PLAN_SUPPLEMENT §1): delta_fb vs N at FIXED c (S7 confounds N with c).
DFBN = dict(c=0.75, Ns=(500, 1000, 2000, 4000), alphas_main=(10.0, 0.3),
            Ns_aux=(500, 2000), alphas_aux=(3.0, 1.0), R=20, hrels=(0.03, 0.01, 0.003), deployed_h=0.03,
            grid="GRID_005", inv="INV_SCALEFREE")
TABLEI = dict(NCH=128, BURN=2000, L=20000, accept_target=0.234,
              regimes=dict(A_strong=dict(d=225, c=0.15, kappa=50.0, r_obs=200.0, eps_rel=0.01),
                           G11=dict(d=225, c=0.15, kappa=50.0, r_obs=4.0, eps_rel=0.01),
                           G12=dict(d=225, c=0.75, kappa=50.0, r_obs=4.0, eps_rel=0.01),
                           G21=dict(d=225, c=0.75, kappa=50.0, r_obs=200.0, eps_rel=0.01),
                           B_weak=dict(d=750, c=0.75, kappa=5.0, r_obs=4.0, eps_rel=0.01)),
              g21_recheck=(dict(lmult=10, seedtag=0, fixed=False), dict(lmult=1, seedtag=1, fixed=False),
                           dict(lmult=1, seedtag=2, fixed=False), dict(lmult=1, seedtag=3, fixed=False),
                           dict(lmult=1, seedtag=0, fixed=True)))

# ---------------------------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------------------------
def workers(requested=None):
    import os
    n = os.cpu_count() or 1
    return max(1, min(n - 1, requested)) if requested else max(1, n - 1)

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


# ---------------------------------------------------------------------------------------------
# GPU execution configuration (M38).  Everything here changes numbers or memory behaviour, so it is
# FROZEN, recorded in the manifest, and never derived from momentary machine state (D25 discipline).
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GPUCfg:
    enabled: bool = True
    # z-points per chunk.  Fixed because chunk size perturbs the last ulps AND the per-point iteration
    # count (stage 2a diagnosis, K=114): torch selects a reduction kernel by shape.  None = size from
    # free memory at run time (recorded as a fallback, not recommended for recorded results).
    chunk_z: int = 65536
    # per-point iteration cap for the GPU kernel; matches the CPU reference (SOLVER.maxit)
    maxit: int = 4000
    # wall-clock budget for one kernel invocation (M38 tail discipline); seconds
    tail_seconds: float = 60.0
    # only feed the kernel to the GPU above this many z-points; below it, the launch overhead and the
    # shape-dependent reduction make the CPU path preferable
    min_points_for_gpu: int = 4096
    # iteration cap for the fp32 rung of the precision ladder.  Measured stage 2b: fp32 cannot meet the
    # 1e-13 pointwise criterion (its eps is 1.2e-7), so all points survive that rung and fp64 finishes
    # in ~2 iterations from the coarse warm start.  Default is 0 = rung DISABLED (a recorded design decision O-2(a),
    # 2026-10-03): the rung rescued 0/60000 points, so keeping it on would burn 20 iterations per call
    # for zero gain.  The knob is retained -- a cheap approximate path must prove itself before it may
    # enter any result; measured, it cannot, so it stays off.  Set > 0 to re-enable a warm-start polish.
    fp32_sweep_maxit: int = 0


GPU = GPUCfg()


# ---------------------------------------------------------------------------------------------
# Experiment-level wall-clock budgets (stage F, a recorded design decision).  v28 §11.3: total = pilot x 2.  These
# are PILOT-CALIBRATED, not hand-typed; an experiment whose budget is not declared here has none, and
# the run must say so rather than imply a bound it does not have.
#
# `expected_seconds` is the extrapolated HEALTHY cost (pilot x cases), which is what the consistency
# check compares against the total.  Do NOT read `tail_seconds x calls` as an expected cost: that is
# the worst case, it normally exceeds the total by design, and the total budget is precisely the
# backstop for when it materialises (see srlab/runbudget.py::contradiction).
# ---------------------------------------------------------------------------------------------
RUN_BUDGETS = {
    # measured pilots: S7 full rerun ~305 s (M14) / ~599 s (M15); EXP-3 check 1018 s for 660 sup rows
    "EXP-3": dict(pilot_seconds=1018.0, pilot_cases=660, cases=660, expected_calls=660),
    "EXP-1": dict(pilot_seconds=360.0, pilot_cases=100, cases=3600, expected_calls=3600),
    "EXP-2": dict(pilot_seconds=85.0, pilot_cases=100, cases=850, expected_calls=850),
}


def run_budget(name, **over):
    """Build the declared RunBudget for an experiment, or None when none is declared.

    Imported lazily so `config` stays importable without the budget module (and so a budget edit can
    never change a number -- it changes only when a run is allowed to stop)."""
    from .runbudget import RunBudget
    spec = dict(RUN_BUDGETS.get(name, {}))
    spec.update(over)
    if not spec:
        return None
    return RunBudget.from_pilot(what=name, tail_seconds=GPU.tail_seconds, **spec)
