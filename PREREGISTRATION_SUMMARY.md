# Pre-registration summary

This is a short English summary of the project's pre-registration record. The full record is kept internally and is available from the authors on request. The paper and its supplementary material are the authoritative description of the method and results; this file only explains how the experimental protocol was fixed and how it changed.

## Protocol

Before any run, the following were frozen (2026-09-25): the target quantities of each experiment, the closed-form predictions they are compared against, sample sizes and repetition counts, seeds, and the acceptance criteria. Every later change was appended as a numbered amendment (M1–M43). The text of an amendment is not edited after registration; results are appended below it.

## Amendments, grouped by theme

**Estimators and statistics (M1–M4, M8, M10–M13).** Recording the per-run count second moment; stable seed derivation (process-randomized string hashing removed); an unbiased variance estimator for Table 1, with the slowest and fastest preconditioned directions added; a global max-|z| test with Šidák correction in place of 48 marginal tests; an audit of the G21 cell in Table 1, where the apparent rejection disappeared at ten times the chain length and is reported as a mixing effect; two bug fixes in the burn-in diagnostic, after which the burn-in budget is reported as thin.

**Supervisor rule (M5, M14, M15, M21, M24).** The supervisor was changed from a stand-alone condition-number threshold, which abstained 127 times at the steps where correction was most needed, to the comparative rule of the paper (26 abstentions); the discretization floor is subtracted before amplification and is refitted on bootstrap samples; inversion failures are classified into typed abstentions; the fallback error was measured against N and found not to decay (S4 of the supplement).

**Regularization (M7, M19).** The Tikhonov weight was changed from a fixed γ₀/N to the scale-free rule γ_rel/R̂_N after the fixed rule degraded the fit in the strong-anisotropy regime. M19 unified all experiments on the scale-free rule.

**Numerical solver (M9, M17, M18, M26, M27, M34, M35).** The activation-zone study was parameterized by bandwidth after the activation boundary was found to move with it; a class of quantile-routine defects was closed across the code base and the affected runs repeated; an exact accelerated quantile routine was validated against the reference (36 of 36 cases); resolution guards and a mass-conservation check were made permanent; the fallback-error solver was hardened with a pointwise convergence criterion.

**Experiments added (M16, M42, M43).** The complete comparison for Proposition 1 (S2 of the supplement); a passive monitor experiment registered with its prediction before computation (the prediction failed and is reported as such, S10); and the Ledoit–Wolf baseline (S5).

**Reproducibility and provenance (M2, M23, M25, M28–M33, M36, M37).** A number-freeze window in which all affected numbers were regenerated in one pass; the deviation ledger, which records for every printed number the raw file, its sha256, the extraction function, and a status; continuous-integration checks on an in-repository data snapshot; and the finding that the BLAS thread count is part of the numerical configuration, so it is pinned per experiment.

**Computing platform (M38–M41).** The forward Stieltjes solver was moved to a GPU kernel with a precision ladder and a bounded wall-clock budget per call. The GPU path was adopted as the source of the paper's numbers only after it reproduced the frozen reference: three experiments bitwise (3600/3600, 850/850, 3960/3960 rows) and the solver acceptance cases to within 6.11 × 10⁻¹⁵ of the reference. An earlier CPU-only platform was retired; its implementation is kept in `reference/` only as the oracle for the equivalence tests and produces none of the paper's numbers.

## Negative results reported

The pre-registered calibration test for the centered statistic failed (about 8% against a 2% tolerance; S2 of the supplement). The monitor prediction of ESS-blind spectral alarms failed (S10). The fallback error does not decay with N (S4). The comparative clause of the supervisor has no true positives on the deployment setting (S6). All four are reported in the paper.
