#!/usr/bin/env python3
"""Typo-repair safety predicate.

The whole point of `repair_typos` is that it may fix a misspelling but must NEVER
substitute a different species. These cases are the ones that decided the rules:

  * a single-token typo (<= 2 edits per token, <= 3 total) is repairable;
  * a *rename* (same animal, new genus) is NOT a typo and must be left alone;
  * `Trapezia wardi -> Trapezia tigrina` is a different crab and must be rejected.

Pure logic: no network, no models, no Qt.
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SPECIES_CFG", str(ROOT / ".test-tmp" / "test_typo_rules.json"))

from repair_typos import tokenwise_ok, levenshtein, epithet_of

FAILS = []


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


print("\n1. ACCEPT: genuine single-token typos")
for a, b in [("Chromi alta", "Chromis alta"),
             ("Chodracanthus exasperatus", "Chondracanthus exasperatus"),
             ("Monodactyus argenteus", "Monodactylus argenteus"),
             ("Atherinopis affins", "Atherinopsis affinis"),
             ("Ovalipe ocellatus", "Ovalipes ocellatus"),
             ("Conu leopardus", "Conus leopardus"),
             ("Anoplopima fimbria", "Anoplopoma fimbria"),
             ("Threagra chalcogramma", "Theragra chalcogramma"),
             ("Hermitripterus americanus", "Hemitripterus americanus")]:
    ok, d = tokenwise_ok(a, b)
    check(f"{a:<28} -> {b:<28} edits={d}", ok)

print("\n2. REJECT: a different species must never be substituted")
for a, b, why in [("Trapezia wardi", "Trapezia tigrina", "a DIFFERENT crab"),
                  ("Chromodoris binza", "Felimida binza", "real rename, genus far"),
                  ("Uca minax", "Minuca minax", "real rename, genus far"),
                  ("Flabellina bicolor", "Samla bicolor", "real rename, genus far"),
                  ("Nembrotha chamberlaini", "Nembrotha purpureolineata",
                   "different nudibranch")]:
    ok, d = tokenwise_ok(a, b)
    check(f"{a:<24} -> {b:<26} edits={d} ({why})", not ok)

print("\n3. The epithet rule keeps same-animal genus moves visible, not silent")
# Same epithet, different genus = plausibly the same animal renamed, so it may be
# CONSIDERED (and then verified); it is never accepted automatically.
for a, b, ep in [("Lolig africana", "Alloteuthis africana", "africana"),
                 ("Lolig sanpaulensis", "Doryteuthis sanpaulensis", "sanpaulensis")]:
    check(f"{a:<24} epithet preserved ({ep})",
          epithet_of(b) == ep and epithet_of(a) == ep,
          f"{epithet_of(a)}/{epithet_of(b)}")

print("\n4. Distance behaves as documented")
check("identical names are 0 apart", levenshtein("Conus", "Conus") == 0)
check("one substitution is 1 apart", levenshtein("Conu", "Conus") == 1)
check("wardi vs tigrina is far apart", levenshtein("wardi", "tigrina") >= 4,
      str(levenshtein("wardi", "tigrina")))

print()
if FAILS:
    print(f"TYPO RULES: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("TYPO RULES: ALL PASS")
