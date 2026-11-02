"""Every tracked worker must be listed in WORKER_ATTRS, or _reap() never clears it.

Twice now a hand-written subset of worker attributes drifted out of date, each time with a
different symptom:

  * `teach_worker` was missing from the subset while its result slot also nulled the reference
    itself, so a QThread was destroyed from inside its own run() and "Teach this photo" aborted
    the process;
  * `worms_worker` was missing after a new worker was added, and because _reap() therefore never
    cleared it, the "already running" guard blocked every later WoRMS lookup — the FIRST lookup
    worked and each one after it silently did nothing.

This test is the guard against a third. It fails as soon as a `self.<name> = SomethingWorker(`
appears that WORKER_ATTRS does not cover.

Pure source inspection: no PyQt, no display, so it runs everywhere.
"""

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

fails = 0


def check(label, ok, detail: object = ""):
    global fails
    if ok:
        print(f"  ok   {label}")
    else:
        fails += 1
        print(f"  FAIL {label}   {str(detail)[:400]}")


src = (REPO / "species_gui.py").read_text()
print("1. the registry exists and is a single literal tuple")
m = re.search(r"^WORKER_ATTRS = \((.*?)\)", src, re.S | re.M)
check("WORKER_ATTRS defined", m is not None)
attrs = tuple(re.findall(r'"([a-z_]+)"', m.group(1))) if m else ()
check("it is not empty", bool(attrs), attrs)
check("no duplicates", len(attrs) == len(set(attrs)), attrs)

print("2. every worker assigned to self is registered")
assigned = sorted(set(re.findall(r"self\.([a-z_]+)\s*=\s*[A-Z]\w*Worker\(", src)))
missing = [a for a in assigned if a not in attrs]
check("no unregistered worker", not missing, missing)
print(f"     assigned: {assigned}")
print(f"     registered: {list(attrs)}")

print("3. the worker classes exist for every assignment")
classes = set(re.findall(r"class ([A-Z]\w*Worker)\(QThread\)", src))
used = set(re.findall(r"self\.([a-z_]+)\s*=\s*([A-Z]\w*Worker)\(", src))
used_cls = {c for _, c in used}
check("every instantiated worker class is defined", used_cls <= classes,
      sorted(used_cls - classes))
print(f"     classes: {sorted(classes)}")

print("4. _reap iterates the registry rather than its own list")
reap = re.search(r"def _reap\(.*?\n    def _track", src, re.S)
check("_reap present", reap is not None)
body = reap.group(0) if reap else ""
check("_reap uses WORKER_ATTRS", "for attr in WORKER_ATTRS" in body)
check("_reap has no hard-coded attribute list of its own",
      not re.search(r'for attr in \("', body), body[:200])

print("5. no slot outside __init__ clears a worker itself")
lines = src.splitlines()


def enclosing(n):
    for i in range(n - 1, -1, -1):
        mm = re.match(r"\s*def (\w+)\(", lines[i])
        if mm:
            return mm.group(1)
    return "?"


pat = re.compile(r"self\.(" + "|".join(attrs) + r")\s*=\s*None")
offenders = [f"line {n} in {enclosing(n)}()" for n, line in enumerate(lines, 1)
             if pat.search(line) and enclosing(n) != "__init__"]
check("only the constructor sets a worker to None", not offenders, offenders)

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
