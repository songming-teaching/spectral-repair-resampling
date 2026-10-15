"""I9 permanent sentinel, source side (M34): every renormalisation in the srlab package must either go
through invariants.normalise() or carry a '# I9:' annotation on the same line stating where its
pre-normalisation gap is checked or why it is definitional.  A bare `p / p.sum()` fails the build --
that is the pattern that hid D23 (m1 = 1.00019 behind an exact unit mass).

Usage: python tools/lint_normalisation.py      (exit 1 on any unannotated site)"""
import os
import re
import sys

PAT = re.compile(r"/\s*[A-Za-z_][\w\.\[\]]*\.sum\(\)|/=\s*[A-Za-z_]\w*\[-1\]|/=\s*[A-Za-z_][\w\.]*\.sum\(\)|"
                 r"/\s*max\(\s*tot|/=\s*len\(|\bg\s*/\s*s\b")
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "srlab")
bad, ok = [], 0
for dp, _dn, fn in os.walk(ROOT):
    for f in fn:
        if not f.endswith(".py"):
            continue
        path = os.path.join(dp, f)
        in_doc = False
        for i, line in enumerate(open(path, encoding="utf-8"), 1):
            n_q = line.count('"""')
            if in_doc or n_q:
                # inside / opening / closing a docstring: prose, not code
                if n_q % 2 == 1:
                    in_doc = not in_doc
                continue
            code = line.split("#", 1)[0]
            if PAT.search(code):
                if "normalise(" in line or "# I9:" in line:
                    ok += 1
                else:
                    bad.append(f"{os.path.relpath(path, ROOT)}:{i}: {line.strip()}")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(f"normalisation sites: {ok} guarded/annotated, {len(bad)} bare")
for b in bad:
    print("  BARE", b)
sys.exit(1 if bad else 0)
