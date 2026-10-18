#!/usr/bin/env python3
"""Test the Lightroom keyword ingester against every export shape."""
import os
os.environ.setdefault("SPECIES_CFG", str(
    __import__("pathlib").Path(__file__).resolve().parent.parent / ".test-tmp" /
    "test_lightroom.json"))
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import lightroom_keywords as lk

fails = []

def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)

# --- the shapes Lightroom can produce -------------------------------------
TABBED = """Animals
\tMarine Life
\t\tCrustaceans
\t\t\tCrabs
\t\t\t\tPortunus pelagicus
\t\t\t\tNeoliomera pubescens
\t\t\tShrimps
\t\t\t\tStenopus hispidus
\t\tNudibranchs
\t\t\tNembrotha chamberlaini
\t\t\tChromodoris annae
\t\tReef Fish
\t\t\tAmphiprion ocellaris
"""

ARROW = """Animals > Marine Life > Crustaceans > Crabs > Portunus pelagicus
Animals > Marine Life > Crustaceans > Crabs > Neoliomera pubescens
Animals > Marine Life > Crustaceans > Shrimps > Stenopus hispidus
Animals > Marine Life > Nudibranchs > Nembrotha chamberlaini
Animals > Marine Life > Nudibranchs > Chromodoris annae
Animals > Marine Life > Reef Fish > Amphiprion ocellaris
"""

PIPE = """Animals|Marine Life|Nudibranchs|Chromodoris annae
Animals|Marine Life|Reef Fish|Amphiprion ocellaris
"""

SPACED = """Animals
  Marine Life
    Crustaceans
      Crabs
        Portunus pelagicus
        Neoliomera pubescens
      Shrimps
        Stenopus hispidus
    Nudibranchs
      Nembrotha chamberlaini
      Chromodoris annae
    Reef Fish
      Amphiprion ocellaris
"""

FLAT = """Portunus pelagicus
Nembrotha chamberlaini
Amphiprion ocellaris
"""

print("A. shape detection")
t = lk.parse(TABBED)
check("tab-indented outline", len(lk.leaves(t)) == 6,
      f"{len(lk.leaves(t))} leaves: {[p[-1] for p in lk.leaves(t)][:3]}")
check("hierarchy preserved", ("Animals", "Marine Life", "Crustaceans", "Crabs",
                              "Portunus pelagicus") in lk.leaves(t))

t2 = lk.parse(ARROW)
check("'>' separated paths", len(lk.leaves(t2)) == 6, f"{len(lk.leaves(t2))} leaves")
check("same tree from a different shape",
      sorted(p[-1] for p in lk.leaves(t2)) == sorted(
          ["Portunus pelagicus", "Neoliomera pubescens", "Stenopus hispidus",
           "Nembrotha chamberlaini", "Chromodoris annae", "Amphiprion ocellaris"]))

t3 = lk.parse(PIPE)
check("'|' separated paths", len(lk.leaves(t3)) == 2, f"{len(lk.leaves(t3))} leaves")

t4 = lk.parse(SPACED)
check("space-indented outline", len(lk.leaves(t4)) == 6, f"{len(lk.leaves(t4))} leaves")

t5 = lk.parse(FLAT)
check("flat list still works", len(lk.leaves(t5)) == 3, f"{len(lk.leaves(t5))} leaves")

print("\nB. the shapes agree with each other")
n_tab = sorted(lk.species_names(lk.parse(TABBED)))
n_arrow = sorted(lk.species_names(lk.parse(ARROW)))
n_space = sorted(lk.species_names(lk.parse(SPACED)))
check("tabbed == arrow", n_tab == n_arrow, f"{len(n_tab)} vs {len(n_arrow)}")
check("tabbed == spaced", n_tab == n_space, f"{len(n_tab)} vs {len(n_space)}")

print("\nC. category nodes are not offered as species")
names = lk.species_names(lk.parse(TABBED))
for bad in ("Animals", "Marine Life", "Crustaceans", "Crabs", "Shrimps",
            "Nudibranchs", "Reef Fish"):
    check(f"'{bad}' excluded", bad not in names)
check("real species kept", "Portunus pelagicus" in names and "Nembrotha chamberlaini" in names,
      f"{len(names)} species names")

print("\nD. NARROWING — the point of keeping the hierarchy")
tree = lk.parse(TABBED)
allnames = lk.species_names(tree)
crust = lk.names_under(tree, "crustacean")
shrimp = lk.names_under(tree, "shrimp")
nudi = lk.names_under(tree, "nudibranch")
fish = lk.names_under(tree, "reef fish")
check("whole list is available", len(allnames) == 6, f"{len(allnames)} names")
check("narrow to crustaceans", len(crust) == 3, f"{len(crust)}: {crust}")
check("narrow to shrimps", shrimp == ["Stenopus hispidus"], str(shrimp))
check("narrow to nudibranchs", nudi == ["Nembrotha chamberlaini", "Chromodoris annae"],
      str(nudi))
check("narrow to reef fish", fish == ["Amphiprion ocellaris"], str(fish))
check("narrowing is a real reduction", len(crust) < len(allnames),
      f"{len(allnames)} -> {len(crust)}")
check("narrowed names are valid candidates (no category words)",
      all(not lk.is_category(n) for n in crust + nudi + fish))

print("\nE. hierarchy summary is printable for inspecting an unknown list")
o = lk.outline(lk.parse(TABBED), max_depth=3)
check("outline lists the tree", "Crustaceans" in o and "Nudibranchs" in o)
check("outline carries leaf counts", "(" in o, o.splitlines()[0] if o else "")

print("\n" + "=" * 70)
print("LIGHTROOM KEYWORD INGESTER: " + ("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
