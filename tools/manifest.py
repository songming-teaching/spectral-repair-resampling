r"""manifest.py -- one manifest per run (REFACTOR_PLAN: "every run ships a manifest").

A manifest is what lets a third party re-walk a "bitwise identical" claim: it pins the code
version, the inputs (by hash), the outputs (by hash), the seeds, the wall time, the environment,
and the M26 I6 resolution indicators.  Figures in deliverables/README.md link to their manifest.

Reproducible PDFs: matplotlib writes the creation time into PDF metadata, so two identical runs
give different PDF bytes.  Setting SOURCE_DATE_EPOCH (the reproducible-builds standard) pins it.
`run_with_manifest` sets it by default, so PDFs become bitwise reproducible too.

Usage (CLI):
    python tools/manifest.py --id <run-id> --cmd "python analyze.py STAGE" \
        --inputs a.json b.npz --outputs figures/fig1.png ... [--stage DIR]
"""
import argparse
import datetime
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical  # noqa: E402  (tools/canonical.py: the ONE volatile-field definition)

# Fixed epoch for reproducible PDF metadata: 2026-09-28 00:00:00 UTC.
DEFAULT_SOURCE_DATE_EPOCH = "1790553600"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_describe(repo):
    try:
        out = subprocess.run(["git", "-C", repo, "describe", "--always", "--dirty", "--tags"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:                                                # noqa: BLE001
        return "unknown"


def env_versions():
    v = dict(python=platform.python_version(), platform=platform.platform())
    for mod in ("numpy", "scipy", "matplotlib"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:                                            # noqa: BLE001
            v[mod] = None
    return v


def dir_digest(path):
    """Recursive content digest of a directory: sha256 over (relative path, file sha256) pairs,
    paths sorted.  Used so a manifest can pin a DIRECTORY input (e.g. `tests/`, `srlab/`) instead of
    crashing on it -- `sha256_file` opens with open(path,"rb") and a directory raises
    PermissionError, which is how the overnight regression task died before running a single test.
    The digest is over content, not mtime, so it is stable across checkouts."""
    h = hashlib.sha256()
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            h.update(rel.encode("utf-8"))
            h.update(sha256_file(full).encode("ascii"))
            total += os.path.getsize(full)
    return h.hexdigest(), total


def files_block(paths, root):
    out = []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(root, p)
        if os.path.isdir(full):
            dig, total = dir_digest(full)
            out.append(dict(path=p, sha256=dig, bytes=total, kind="dir"))
        elif os.path.exists(full):
            blk = dict(path=p, sha256=sha256_file(full), bytes=os.path.getsize(full), kind="file")
            # The RAW sha above is the primary identity and is never removed.  For JSON outputs we
            # additionally record the CONTENT sha (tools/canonical.py): the same numbers with the
            # machine-dependent duration fields dropped and path echoes folded.  Two runs whose
            # NUMBERS agree but whose wall clocks differ then verify as `numeric` instead of
            # failing outright -- which is what makes the paper's reviewer path satisfiable, since
            # every row file records a duration.  Entries without a content form (non-JSON, or
            # unparseable) simply omit the key, and `verify` falls back to the raw comparison.
            if p.lower().endswith(".json"):
                cs = canonical.content_sha256_file(full)
                if cs is not None:
                    blk["content_sha256"] = cs
            out.append(blk)
        else:
            out.append(dict(path=p, sha256=None, bytes=None, kind=None, missing=True))
    return out


def collect_cap(outputs, stage):
    """M26: gather every I6 cap record ({"policy", "worst_dx_over_eta", "exceeded", ...}) found anywhere
    in the run's JSON outputs, so the indicator lands in the manifest automatically -- not only in a
    return value nobody reads."""
    recs = []

    def walk(o):
        if isinstance(o, dict):
            if "worst_dx_over_eta" in o and "exceeded" in o:
                recs.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for p in outputs:
        full = p if os.path.isabs(p) else os.path.join(stage, p)
        if full.endswith(".json") and os.path.exists(full):
            try:
                with io.open(full, encoding="utf-8") as fh:
                    walk(json.load(fh))
            except (ValueError, OSError):
                pass
    if not recs:
        return dict(quantile_calls=0, note="no law_quantiles call recorded in the outputs")
    return dict(quantile_calls=len(recs), exceeded=sum(1 for r in recs if r["exceeded"]),
                worst_dx_over_eta=max(r["worst_dx_over_eta"] for r in recs),
                policies=sorted({r.get("policy", "?") for r in recs}))


def collect_run_stats(outputs, stage):
    """M35: retry depth and reason codes, gathered automatically from the run's JSON outputs:
    straggler escalation depths (depth0/1/2), typed solver failures, and abstention reason codes."""
    esc = dict(depth0=0, depth1=0, depth2=0); typed = 0; reasons = {}

    def walk(o):
        nonlocal typed
        if isinstance(o, dict):
            se = o.get("straggler_escalation")
            if isinstance(se, dict):
                for k in esc:
                    esc[k] += int(se.get(k, 0))
            if "typed_failure" in o:
                typed += 1
            r = o.get("abstain_reason")
            if r:
                reasons[r] = reasons.get(r, 0) + 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for p in outputs:
        full = p if os.path.isabs(p) else os.path.join(stage, p)
        if full.endswith(".json") and os.path.exists(full):
            try:
                with io.open(full, encoding="utf-8") as fh:
                    walk(json.load(fh))
            except (ValueError, OSError):
                pass
    return dict(straggler_escalation=esc, typed_solver_failures=typed, abstain_reasons=reasons)


def resolve_interpreter(cmd):
    """INTERPRETER DISCIPLINE (fixed 2026-10-03).  The command is executed with `shell=True`, so a bare
    "python x.py" is resolved by the shell against PATH.  On this machine that is the CPU-ONLY torch
    build, and a GPU run launched that way produces `cpu-fallback:no_cuda` for every case while its
    numbers still look entirely plausible -- the failure is silent and the timing record is worthless.
    A night of D20 work was at risk of exactly this.

    Rewriting the token HERE, in the one function that actually spawns the process, covers every caller
    at once (the overnight runner, CI, and a hand-typed manifest invocation).  `sys.executable` is the
    interpreter this manifest process itself runs under, which is the one the caller selected.

    Commands that already name an interpreter explicitly are left untouched, and `python -c`/`-m` are
    rewritten the same way (only the leading token matters)."""
    import sys
    for token in ("python.exe ", "python "):
        if cmd.startswith(token):
            return sys.executable + " " + cmd[len(token):]
    return cmd


def is_backend_path(v):
    """True iff `v` is a FORWARD-MAP backend path (srlab/forward.py:47) -- the authoritative
    vocabulary, spelled out here rather than pattern-matched loosely: "gpu" | "cpu-forced" |
    "cpu-fallback:<reason>" | "mixed:<paths>".

    WHY A VALUE ALLOWLIST AND NOT A KEY MATCH.  Four different things in this codebase are called
    `path`, and only one of them is a backend:

      1. backend          "gpu" / "cpu-fallback:no_cuda" / "mixed:..." / "cpu-forced"   <- wanted
      2. EXP-3 method     "corrected" / "fallback:no_correction" / "fallback:forced_schedule"
                          (srlab/experiments/exp3.py:82,123,145,148) -- a supervisor verdict, not a
                          backend.  Note it sits ONE HYPHEN away from a real value: "fallback:x"
                          versus "cpu-fallback:x", which is what makes a prefix test unsafe.
      3. kernel rung      "cpu" / "gpu" / "gpu+fp64"  (tail.levels[].path) -- per-rung detail
      4. filesystem       manifests' `outputs[].path` -- a filename

    Keying on the name `path` therefore sweeps in 2 and 4, and a `startswith("fallback")` test
    sweeps in 2 while looking like it excludes it.  Matching the VALUE against the producer's own
    vocabulary is the only test that separates them."""
    if not isinstance(v, str):
        return False
    return v in ("gpu", "cpu-forced") or v.startswith("cpu-fallback:") or v.startswith("mixed:")


def backend_paths(obj):
    """Every backend path string recorded inside `obj` -- ONE definition, two consumers
    (this module's collect_backend, and tools/build_ledger.py's per-entry `path`).

    A first version keyed on the name `path` and additionally skipped the `tail`/`levels` subtree to
    avoid the kernel's rung vocabulary (case 3 above); it recorded a 6/6-GPU run as
    ["cpu","gpu","gpu+fp64"].  Skipping the subtree fixed that one and left case 2 -- EXP-3's row
    `path`, which is a supervisor verdict -- still bleeding in as ["corrected","fallback:..."].
    Both are now excluded by the value test, so the subtree skip is a belt to the allowlist's
    braces.  (Fourth vocabulary collision found on this field, 2026-10-03; each of the first three
    also looked right until it was run against real data.)

    Nothing is inferred: a source that records no backend yields an empty set, and the caller
    reports `null` -- never a default of "cpu", which would put words in the data's mouth."""
    KEYS = ("path", "fast_path", "fast_path_cpu", "paths_seen")
    SKIP_SUBTREES = ("tail", "levels")          # the kernel's internal escalation ladder (case 3)
    acc = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in KEYS:
                    if is_backend_path(v):
                        acc.add(v)
                    elif isinstance(v, list):
                        acc.update(x for x in v if is_backend_path(x))
                if k not in SKIP_SUBTREES:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    return acc


def collect_backend(outputs, stage):
    """O-3 / M40 boundary 2: record the backend path(s) IN THE MANIFEST, so a run's provenance does
    not depend on someone reading its output rows afterwards.

    The recorded decision requires it at this layer for a specific reason: "every fallback must be recorded in the manifest and flagged in the ledger status".
    A CPU fallback that is only visible inside a result file is visible to whoever thinks to open
    that file; a fallback recorded in the manifest is visible to the audit that walks manifests.
    Golden regression checks this field.

    Returns None when the run's outputs record no backend at all -- an honest "not recorded".
    `sources` names the output files the paths came from, so the claim can be re-walked."""
    found, sources = set(), []
    for p in outputs:
        full = p if os.path.isabs(p) else os.path.join(stage, p)
        if full.endswith(".json") and os.path.exists(full):
            try:
                with io.open(full, encoding="utf-8") as fh:
                    got = backend_paths(json.load(fh))
            except (ValueError, OSError):
                continue
            if got:
                found |= got
                sources.append(os.path.relpath(full, stage).replace("\\", "/"))
    if not found:
        return None
    return dict(paths=sorted(found), sources=sources)


def collect_budget(outputs, stage):
    """stage F (a recorded design decision): the experiment-level budget block, if the run recorded one.

    A run's total-budget state (`srlab.runbudget.RunBudget.manifest_block()`) lands in its output
    JSON, so it reaches the manifest the same way `cap` does -- gathered from the outputs, not passed
    by a caller who might forget.  Returns None when the run recorded no budget: an experiment that
    does not declare one is not thereby unbounded, but the manifest should not invent a number for
    it either.  `status == "budget_exceeded"` here is what tells a reader the run was CUT OFF and its
    output is partial (see RunBudget.partial())."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            if "budget_s" in o and "status" in o and ("eta_s" in o or "pct_used" in o):
                found.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for p in outputs:
        full = p if os.path.isabs(p) else os.path.join(stage, p)
        if full.endswith(".json") and os.path.exists(full):
            try:
                with io.open(full, encoding="utf-8") as fh:
                    walk(json.load(fh))
            except (ValueError, OSError):
                pass
    if not found:
        return None
    return found[0] if len(found) == 1 else dict(runs=found)


def run_with_manifest(run_id, cmd, inputs, outputs, stage, repo, manifest_dir,
                      seeds=None, cap=None, source_date_epoch=DEFAULT_SOURCE_DATE_EPOCH, notes=None):
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = source_date_epoch
    env.setdefault("PYTHONHASHSEED", "0")
    t0 = time.time()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd_as_given = cmd
    cmd = resolve_interpreter(cmd)
    proc = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    wall = time.time() - t0
    man = dict(
        run_id=run_id,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        code_version=git_describe(repo),
        command=cmd,
        # The command AS GIVEN, plus the interpreter actually used.  `command` alone shows the
        # rewritten form, so a reader could not tell what was asked for; recording both keeps the
        # audit trail honest about what ran and why (interpreter discipline, 2026-10-03).
        command_as_given=cmd_as_given,
        interpreter=sys.executable,
        returncode=proc.returncode,
        wall_seconds=round(wall, 3),
        environment=env_versions(),
        environment_block={k: env.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                  "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED", "SOURCE_DATE_EPOCH",
                                                  "SRLAB_EXP", "SR_FULL")},
        env_pins=dict(SOURCE_DATE_EPOCH=source_date_epoch, PYTHONHASHSEED=env["PYTHONHASHSEED"],
                      # D25: the BLAS thread count changes LAPACK results in the last ulps and, through a flat
                      # inversion optimum, Table I chain statistics -- it is part of the numerical configuration
                      **{k: env.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}),
        seeds=seeds,
        inputs=files_block(inputs, stage),
        outputs=files_block(outputs, stage),
        cap=cap if cap is not None else collect_cap(outputs, stage),   # M26: mandatory, gathered automatically
        solver_stats=collect_run_stats(outputs, stage),                  # M35: retry depth + reason codes
        budget=collect_budget(outputs, stage),                           # stage F: total-budget state + ETA
        backend=collect_backend(outputs, stage),                         # O-3/M40: the path the run took
        notes=notes,
        stderr_tail=proc.stderr[-2000:] if proc.stderr else "",
        stdout_tail=proc.stdout[-4000:] if proc.stdout else "",   # night-run fix L4: progress lines were dropped
    )
    os.makedirs(manifest_dir, exist_ok=True)
    with io.open(os.path.join(manifest_dir, f"{run_id}.stdout.log"), "w", encoding="utf-8") as fh:
        fh.write(proc.stdout or "")
    path = os.path.join(manifest_dir, f"{run_id}.json")
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=1)
    return man, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--outputs", nargs="*", default=[])
    ap.add_argument("--stage", default=".")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--manifest-dir", default="manifests")
    args = ap.parse_args()
    man, path = run_with_manifest(args.id, args.cmd, args.inputs, args.outputs, args.stage,
                                  args.repo, args.manifest_dir)
    print(f"manifest -> {path}  (returncode {man['returncode']}, {man['wall_seconds']} s)")
    return man["returncode"]


if __name__ == "__main__":
    sys.exit(main())
