r"""python -m srlab -- the reviewer path promised in REFACTOR_PLAN.md:3 and the paper's §IX.H.

    python -m srlab --help
    python -m srlab list                         # the registered experiments and figure steps
    python -m srlab run all [--out-root DIR]     # every experiment, then every figure step
    python -m srlab run exp1 [--out-root DIR]    # one experiment
    python -m srlab verify [--out-root DIR]      # sha256 of re-run outputs vs the registered manifests

The registry below is NOT hand-written: each command is the `command_as_given` of the manifest that
produced the window result the paper carries (manifests/window-*.json, fig-*.json).  Only the
output directory is redirected, to `--out-root` (default results/rerun), so a reviewer's re-run can
never overwrite the registered products; `verify` then compares the re-run files with the sha256
recorded in those manifests.

Interpreter: every step runs under THIS interpreter (sys.executable), never a bare `python` from
PATH -- the night_run.py lesson (a PATH python on this machine is the CPU build).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (manifest id, module, args with {out} for the output dir, registered output dir)
#
# This dict is the FALLBACK, used only when a manifest is missing.  The authoritative registry is
# built by `load_registry()` from each manifest's own `command_as_given`, because the hand-written
# copy DRIFTED: on 2026-10-11 the reviewer path was found to be missing `--workers 8` for dfb_n and
# exp1_lw, so it ran under the default 31 processes instead of the registered 8, contended for the
# GPU, and pushed one cell past the 60 s tail budget into a registered abstention.  The rerun then
# failed to reproduce a run that was in fact perfectly reproducible.  A reviewer path that silently
# changes the numerical configuration (thread/worker count is part of that configuration, D25/M37)
# cannot support a reproducibility claim, so the commands are now read from the manifests rather
# than restated here.
EXPERIMENTS = {
    "exp1":   ("window-m19-exp1",  "srlab.experiments.exp1",   "--rule m19 --out {out}", "results/window/exp1_m19"),
    "exp2":   ("window-m19-exp2",  "srlab.experiments.exp2",   "--rule m19 --out {out}", "results/window/exp2_m19"),
    "exp3":   ("window-m19-exp3",  "srlab.experiments.exp3",   "--rule m19 --out {out}", "results/window/exp3_m19"),
    "tableI": ("window-m20-tableI", "srlab.experiments.tableI",
               "--regimes A_strong,G11,G12,G21,B_weak --seeds m33 --out {out} --workers 4", "results/window/tableI_m20"),
    "s7":     ("window-s7-gpu",    "srlab.experiments.s7",     "--mode window --out {out} --workers 1", "results/window/s7"),
    "exp1_lw": ("window-m43-exp1-lw", "srlab.experiments.exp1_lw",
                "--ref results/window/exp1_m19/exp1_rows.json --out {out} --workers 8", "results/window/exp1_lw"),
    "dfb_n":  ("window-m24-dfb-n", "srlab.experiments.dfb_n",  "--out {out} --workers 8", "results/window/dfb_N"),
}


def _parse_registered_command(cmd):
    """Split a manifest's `command_as_given` into (module, args) with the output dir templated.

    Returns None when the shape is not `... -m <module> <args>` or when no `--out` value is present.
    The `--out` value seen here IS the registered output directory, so both the template and
    `reg_dir` come from the one source and cannot disagree.
    """
    if not isinstance(cmd, str):
        return None
    parts = cmd.split()
    if "-m" not in parts:
        return None
    i = parts.index("-m")
    if i + 1 >= len(parts):
        return None
    module, args = parts[i + 1], parts[i + 2:]
    if not module.startswith("srlab."):
        return None
    reg_dir = None
    for j, a in enumerate(args):
        if a == "--out" and j + 1 < len(args):
            reg_dir = args[j + 1]
            args[j + 1] = "{out}"
            break
    if reg_dir is None:
        return None
    return module, " ".join(args), reg_dir


def load_registry():
    """`EXPERIMENTS`, but each entry's command taken from the manifest that produced the result.

    Falls back to the static entry (and says so) when a manifest is missing or does not carry a
    parseable `command_as_given`.  A fallback is reported rather than silent: a reviewer must be able
    to tell that a command came from this file rather than from the recorded run.
    """
    out = {}
    for name, (mid, module, args, reg_dir) in EXPERIMENTS.items():
        mp = os.path.join(REPO, "manifests", mid + ".json")
        try:
            with open(mp, encoding="utf-8") as fh:
                man = json.load(fh)
        except (OSError, ValueError):
            out[name] = (mid, module, args, reg_dir, "fallback: no manifest")
            continue
        parsed = _parse_registered_command(man.get("command_as_given") or man.get("command"))
        if not parsed:
            out[name] = (mid, module, args, reg_dir, "fallback: unparseable command")
            continue
        m2, a2, r2 = parsed
        src = "manifest"
        if m2 != module:
            # The manifest names a different module than this file expects: prefer the manifest (it
            # is the record of what actually ran) but make the disagreement visible.
            src = f"manifest (module {m2} != expected {module})"
        out[name] = (mid, m2, a2, r2, src)
    return out

# figure steps read the assembled stage; they write into it, so they run on a COPY under --out-root
FIGURES = [
    "scripts/analyze.py",
    "scripts/tableI_render.py",
    "scripts/s1_countlaw.py",
    "scripts/s3_analyze.py",
    "scripts/s4_sensitivity.py",
    "scripts/s5_phasediagram.py",
    "scripts/s6_burnin.py",
    "scripts/monitor_figure.py",
]


def _run(argv, log):
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(argv, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return rc, time.time() - t0


def run_experiment(name, root, registry=None):
    mid, mod, args, _, _src = (registry or load_registry())[name]
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)
    argv = [sys.executable, "-m", mod] + args.format(out=out).split()
    rc, dt = _run(argv, os.path.join(root, f"{name}.log"))
    print(f"  {name:8s} rc={rc}  {dt / 60:6.1f} min  -> {out}", flush=True)
    return rc


def run_figures(root):
    import shutil
    # Figures are drawn from the archived data package (Zenodo), laid out as data/, figures/, results/.
    # Set SRLAB_DATA to its location; the package is copied so that the figure scripts never write
    # into the archived copy.
    src = os.environ.get("SRLAB_DATA")
    if not src or not os.path.isdir(src):
        print("  figures: set SRLAB_DATA to the unpacked data package (see README)", flush=True)
        return 1
    stage = os.path.join(root, "stage")
    if not os.path.isdir(stage):
        shutil.copytree(src, stage)
    rcs = []
    for script in FIGURES:
        rc, dt = _run([sys.executable, script, stage],
                      os.path.join(root, os.path.basename(script) + ".log"))
        print(f"  {os.path.basename(script):22s} rc={rc}  {dt:6.1f} s", flush=True)
        rcs.append(rc)
    return max(rcs) if rcs else 0


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root):
    """Compare a re-run against the registered manifests.  THREE states, never a blanket verdict:

        bitwise  raw sha256 identical                              (strongest claim)
        numeric  raw differs, canonical sha256 identical           (only wall-clock moved)
        DIFFERS  canonical sha256 differs                          (a real change -> investigate)

    The `numeric` state exists because every row file records a duration (`sec`), so a whole-file
    hash can never reproduce; tools/canonical.py is the ONE place the machine-dependent fields are
    defined, and it is shared with the manifest writer, so both sides agree by construction.  A file
    with no canonical form (non-JSON) gets no escape hatch: only `bitwise` or `DIFFERS`.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(REPO, "tools"))
    import canonical  # noqa: E402

    bad = 0; checked = 0; soft = 0
    for name, (mid, _, _, reg_dir) in EXPERIMENTS.items():
        mp = os.path.join(REPO, "manifests", mid + ".json")
        if not os.path.isfile(mp):
            print(f"  {name:8s} no registered manifest ({mid}) -- skipped")
            continue
        with open(mp, encoding="utf-8") as fh:
            outs = [o for o in json.load(fh).get("outputs", []) if o.get("kind") == "file"]
        for o in outs:
            rel = os.path.relpath(o["path"], reg_dir)
            p = os.path.join(root, name, rel)
            checked += 1
            if not os.path.isfile(p):
                print(f"  {name:8s} MISSING  {rel}"); bad += 1
            elif _sha(p) == o["sha256"]:
                print(f"  {name:8s} bitwise  {rel}")
            else:
                reg_canon = o.get("content_sha256")
                run_canon = canonical.content_sha256_file(p) if reg_canon else None
                if reg_canon and run_canon == reg_canon:
                    # Same numbers, different wall clocks.  Reported as its own state, counted
                    # apart from `bad`, and it is NOT called bitwise.
                    print(f"  {name:8s} numeric  {rel}  (numbers identical; only machine-dependent "
                          f"fields {sorted(canonical.VOLATILE_KEYS)}·path echoes differ)")
                    soft += 1
                else:
                    print(f"  {name:8s} DIFFERS  {rel}  (see ledger/ for registered tolerances)")
                    bad += 1
    print(f"verify: {checked - bad - soft}/{checked} bitwise, {soft} numerically identical "
          f"(wall-clock only), {bad} differ")
    return 0 if bad == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m srlab", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list registered experiments and figure steps")
    r = sub.add_parser("run", help="re-run one experiment, or all of them followed by the figures")
    r.add_argument("what", choices=["all", "figures", *EXPERIMENTS])
    r.add_argument("--out-root", default=os.path.join("results", "rerun"))
    v = sub.add_parser("verify", help="compare a re-run with the registered manifest sha256")
    v.add_argument("--out-root", default=os.path.join("results", "rerun"))
    a = ap.parse_args(argv)

    if a.cmd == "list":
        reg = load_registry()
        for k, (mid, mod, args, rdir, src) in reg.items():
            print(f"{k:8s} {mod} {args}   [registered: {rdir}, manifest {mid}, args from {src}]")
        for s in FIGURES:
            print(f"figure   {s} <stage>")
        return 0
    root = os.path.join(REPO, a.out_root) if not os.path.isabs(a.out_root) else a.out_root
    os.makedirs(root, exist_ok=True)
    if a.cmd == "verify":
        return verify(root)
    reg = load_registry()
    names = list(reg) if a.what == "all" else ([] if a.what == "figures" else [a.what])
    rc = 0
    for n in names:
        rc = max(rc, run_experiment(n, root, reg))
    if a.what in ("all", "figures"):
        rc = max(rc, run_figures(root))
    return rc


if __name__ == "__main__":
    sys.exit(main())
