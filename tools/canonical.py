r"""canonical.py -- the ONE definition of "which fields are machine-dependent", used by both the
manifest writer (tools/manifest.py) and the reviewer path (srlab/__main__.py verify).

WHY THIS EXISTS
---------------
The row files carry wall-clock fields (`sec`, `sec_invert`, ...).  Two independent runs of the same
experiment produce NUMERICALLY IDENTICAL rows -- verified 2026-10-11 for EXP-1 (3600 rows), the LW
baseline (900 rows) and the M24 delta_fb sweep (240 rows): after removing the volatile keys, the
canonical forms are equal, and no other field moves.  But a manifest that hashes the WHOLE FILE can
never reproduce, because the timings differ run to run.

The ledger layer has always known this: ledger/tolerance_classes.json registers a `machine_timing`
class whose rule is "wall-clock timings are machine- and load-dependent, so their regression status
is AT BEST `tol`, never bitwise".  The manifest / verify layer simply had not caught up with that
recorded decision, so the paper's reproducibility statement ("verify compares the sha256 recorded in those
manifests") was unsatisfiable for any run that reported a duration.

WHAT IT DOES NOT DO
-------------------
It does not make a non-reproducible run look reproducible.  Manifests keep the RAW sha256 as well;
`verify` reports THREE states, and only claims `bitwise` when the raw bytes match:

    bitwise      raw sha256 identical                       (strongest)
    numeric      raw differs, canonical sha256 identical    (only timing fields moved)
    DIFFERS      canonical sha256 differs                   (a real change -> investigate)

VOLATILITY IS SEMANTIC, NOT COSMETIC
------------------------------------
`VOLATILE_KEYS` is a FROZEN registry.  Adding a key here weakens what `numeric` proves, so the list
is deliberately short and every entry is a wall-clock or duration measurement.  Never add a key
merely because it is inconvenient to reproduce: that would hide a real regression behind a `numeric`
label, which is the failure mode this module exists to prevent.
"""

import json
import os

# Machine-dependent fields.  FROZEN: see the module docstring before extending.
# A key is volatile iff it measures elapsed time / wall clock, nothing else.
VOLATILE_KEYS = frozenset({
    "sec",          # per-row or per-call duration (exp1, exp1_lw, s7, ...)
    "sec_invert",   # the inversion stage's share (dfb_n)
    "sec_dfb",      # the delta_fb stage's share (dfb_n)
    "wall_sec",     # whole-run duration (tableI rows)
    "wall_seconds", # whole-run duration (manifest / summary level)
    "elapsed",      # generic elapsed
    "eta_sec",      # projected remaining time (runbudget block)
})


def strip_volatile(obj):
    """Recursively drop every VOLATILE_KEYS entry from dicts.  Lists are rebuilt elementwise."""
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


ROOT_TOKEN = "<ROOT>"


def _normalise_path(s, roots):
    """Replace any `roots` prefix by ROOT_TOKEN and use '/' separators.

    A file that echoes WHERE it was produced (`source: results/window/dfb_N/dfb_N_rows.json`) is
    recording provenance, not content: a reviewer who re-runs into their own output directory should
    not see a content difference.  Both the registered root and the re-run root map to the same
    token, so the two bodies compare equal while a genuinely different file name still shows up.

    BOTH addressing forms are folded.  The same file can be named relatively
    (`results/window/dfb_N/x.json`) or absolutely (`C:/.../results/window/dfb_N/x.json`) depending on
    which side is doing the naming, and an identity that changed with the spelling would be useless:
    a hash computed from an absolute path would never match one computed from a relative path, even
    for a byte-identical file (this exact mismatch was observed 2026-10-11).  Every candidate form of
    the string is therefore tried against every root, both sides reduced to absolute first.  Roots are
    tried longest-first so a nested root is not shadowed by its parent.
    """
    if not isinstance(s, str) or not roots:
        return s
    flat = s.replace("\\", "/")
    cands = {flat}
    try:
        cands.add(os.path.abspath(s).replace("\\", "/"))
    except (OSError, ValueError):
        pass
    rootfs = set()
    for r in roots:
        try:
            rootfs.add(os.path.abspath(r).replace("\\", "/").rstrip("/"))
        except (OSError, ValueError):
            continue
    for rf in sorted(rootfs, key=len, reverse=True):
        if not rf:
            continue
        for c in cands:
            if c == rf or c.startswith(rf + "/"):
                return ROOT_TOKEN + c[len(rf):]
    return flat


def normalise(obj, roots=()):
    """strip_volatile + path-echo normalisation + '\\' -> '/' for every string."""
    obj = strip_volatile(obj)
    roots = tuple(roots or ())

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str):
            return _normalise_path(o, roots)
        return o

    return walk(obj)


def canonical_json(obj):
    """A deterministic serialisation: sorted keys, fixed separators, NaN/inf spelled out.

    Sorting matters because two producers can emit the same mapping in different insertion order.
    NaN is spelled as a string because `json.dumps(nan)` emits the non-standard literal `NaN`, which
    a strict parser rejects; and in these files NaN is a legitimate value (a condition number that
    could not be formed), not an error -- it must compare equal to itself across runs.
    """
    import json

    def norm(o):
        if isinstance(o, dict):
            return {str(k): norm(v) for k, v in o.items()}
        if isinstance(o, list):
            return [norm(v) for v in o]
        if isinstance(o, float):
            if o != o:                      # NaN
                return "__NaN__"
            if o == float("inf"):
                return "__Inf__"
            if o == float("-inf"):
                return "__-Inf__"
        return o

    return json.dumps(norm(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def canonical_bytes(obj):
    return canonical_json(obj).encode("utf-8")


def content_sha256(obj, roots=()):
    """The ONE content identity: volatile fields dropped, path echoes folded, keys sorted.

    There is deliberately no way to ask for a "canonical" hash that KEEPS the volatile fields --
    that hash would differ between two runs of this very module and would be worthless for the
    purpose it exists for.  The raw byte identity is a separate thing and lives in the manifest's
    `sha256` field; this is the fallback that says "the numbers matched".
    """
    import hashlib
    return hashlib.sha256(canonical_bytes(normalise(obj, roots))).hexdigest()


def content_sha256_file(path, roots=None):
    """`content_sha256` for a JSON file.  None when the file is not parseable JSON -- the caller
    must then rely on the raw sha alone: a non-JSON output gets no `numeric` escape hatch.

    `roots` defaults to the file's OWN directory.  An output that echoes where it was written
    (`"source": "results/window/dfb_N/dfb_N_rows.json"`) is recording provenance, and a reviewer who
    re-runs into a different directory should still see a content match; folding the file's own
    directory makes the writer and the verifier agree by construction, since both sides fold the
    identical basename tail.  Pass an explicit `roots` to fold additional prefixes.
    """
    import io
    import json
    if roots is None:
        roots = (os.path.dirname(path),)
    try:
        with io.open(path, encoding="utf-8") as fh:
            return content_sha256(json.load(fh), roots)
    except (ValueError, OSError):
        return None
