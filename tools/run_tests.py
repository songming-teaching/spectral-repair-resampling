r"""run_tests.py -- run the srlab test suite the way CI does, without pytest.

WHY THIS EXISTS.  The overnight queue used to call `pytest`, which is not installed in the CUDA
virtualenv (`a CUDA-enabled environment`) -- the interpreter the night runner must use for the
GPU path.  `pytest` would fail with "No module named pytest" and the regression step would be silently
absent from the night's work.  The repository's own convention (see .github/workflows/tests.yml) is
that every tests/*.py is a standalone script that prints [PASS]/[FAIL] lines and exits non-zero on
failure; this runner follows exactly that convention, so it works in any interpreter that can run the
suite at all.

Each test runs in its OWN subprocess: one crashing test (a segfault in torch, an OOM) must not take
the rest down, and the per-test return code is recorded.

Usage:
  python tools/run_tests.py                 # all tests, default timeout per test
  python tools/run_tests.py --only test_laws test_solver
  python tools/run_tests.py --timeout 3600
Exit code 0 only if every selected test passed.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
OUT = os.path.join(ROOT, "results", "checks", "test_suite.json")

# Order: cheap and fundamental first, so a late failure still leaves the early evidence on disk.
# `equivalence.py` is a library used by other tests, not a test, and is skipped.
SKIP = {"equivalence.py"}


def discover():
    return sorted(f for f in os.listdir(TESTS) if f.startswith("test_") and f.endswith(".py"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="test module names without .py, e.g. --only test_laws")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per test")
    a = ap.parse_args(argv)

    names = [f[:-3] for f in discover() if f not in SKIP]
    if a.only:
        want = {n.replace(".py", "") for n in a.only}
        names = [n for n in names if n in want]

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    rows, failed = [], []
    for n in names:
        path = os.path.join("tests", n + ".py")
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, path], cwd=ROOT, env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=a.timeout)
            rc, out, err, rc_note = p.returncode, p.stdout, p.stderr, ""
        except subprocess.TimeoutExpired as exc:
            rc, rc_note = 124, f"timeout after {a.timeout:.0f}s"
            out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            err = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        wall = time.time() - t0
        tail = "\n".join((out or "").strip().splitlines()[-6:])
        # A SKIP IS NOT A PASS.  The GPU tests return 0 after printing "skipped: no CUDA device" on a
        # runner without one, so counting rc==0 as PASS would report green for work that never ran --
        # the same class of sediment this runner exists to clear.
        #
        # The test files print their own [PASS]/[FAIL]/[SKIP] lines, one per sub-test, so a file is
        # SKIP only when it did NO work: no [PASS] and no [FAIL], but a skip marker.  A file that ran
        # four sub-tests and skipped a fifth is PASS-with-a-note, not SKIP -- an earlier version of
        # this check looked only for the word "skipped" and wrongly reported test_forward_oracle (which
        # runs real coverage in both modes) as entirely skipped, understating what CI actually covers.
        body = (out or "")
        low = body.lower()
        n_sub_pass = low.count("[pass]")
        n_sub_fail = low.count("[fail]")
        skip_note = ""
        if "skipped" in low:
            for line in body.splitlines():
                if "skipped" in line.lower():
                    skip_note = line.strip()[:120]
                    break
        skipped = rc == 0 and n_sub_pass == 0 and n_sub_fail == 0 and "skipped" in low
        status = "SKIP" if skipped else ("PASS" if rc == 0 else "FAIL")
        rows.append(dict(test=n, rc=rc, status=status, wall_s=round(wall, 1), note=rc_note or skip_note,
                         tail=tail, stderr_tail=(err or "")[-500:] if rc else ""))
        if rc != 0:
            failed.append(n)
        print(f"[{status}] {n}  ({wall:.1f}s)" + (f"  {rc_note or skip_note}" if (rc_note or skip_note) else ""),
              flush=True)

    n_skip = sum(1 for r in rows if r["status"] == "SKIP")
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(dict(interpreter=sys.executable, python=sys.version.split()[0],
                       n=len(rows), n_pass=n_pass, n_skipped=n_skip, failed=failed, rows=rows),
                  fh, indent=1)
    print(f"\n{n_pass} passed, {n_skip} skipped, {len(failed)} failed (of {len(rows)})"
          + (f"; FAILED: {failed}" if failed else "") + f"  -> {OUT}")
    if failed:
        for r in rows:
            if r["rc"] != 0:
                print(f"\n--- {r['test']} (rc={r['rc']}) ---\n{r['tail']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
