#!/usr/bin/env python3
"""The parse stage must survive the keyword export's real shape.

The export is CR-only (\r line endings, no \n anywhere), tab-indented by depth, with
leaves written as "common name: common name: Genus species". Every one of those details
has bitten a naive parser:

  - reading it as \n-delimited yields ONE line, so everything becomes one node
  - splitting the leaf on ":" and taking [0] returns a common name, not a binomial
  - leaves with no common names ("Genus species" alone) must not vanish
  - a colon inside a common name must not shift the epithet

No network, no models, no Qt — this runs in CI.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightroom_keywords as lk

checks = failed = 0


def check(label: str, got, want) -> None:
    global checks, failed
    checks += 1
    if got == want:
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}\n          got  {got!r}\n          want {want!r}")


# A miniature export with the same structure as MarineLife Keywords 1.0, including the
# awkward cases. Joined with \r only — that is the whole point.
LEAVES = [
    "Marine Life",
    "\tAlgae",
    "\t\tBrown Algae",
    "\t\t\tBrown Algae: Brown Seaweeds: Phaeophyta",
    "\t\t\t\tbladder wrack: rockweed: Fucus vesiculosus",
    "\t\t\t\tblistered saucer leaf algae: Turbinaria turbinata",
    "\t\tRed Algae",
    "\t\t\tAgarum fimbriatum",                       # no common names at all
    "\t\t\tdead man's fingers: Codium fragile",       # apostrophe + space
    "\tAnimals",
    "\t\tNudibranchs",
    "\t\t\tChromodoris annae",                        # bare binomial
    "\t\t\tAnna's chromodoris: Chromodoris annae",    # duplicate species, other name
]
text = "\r".join(LEAVES) + "\r"

# The file must really be CR-only: a parser that got lucky with \n would pass otherwise.
check("fixture is CR-only", "\n" in text, False)

tree = lk.parse(text)
recs = lk.species_entries(tree)

# 5 unique species across 7 species-level leaves: "Chromodoris annae" is listed twice
# (once bare, once with a common name) and must collapse to a single record.
check("unique species out of the leaves", len(recs), 5)
by_name = {r["scientific"]: r for r in recs}

check("binomial extracted from a leaf with common names",
      "Fucus vesiculosus" in by_name, True)
check("common names split off the binomen",
      by_name["Fucus vesiculosus"]["common"], ["bladder wrack", "rockweed"])
check("leaf with no common names is kept",
      "Agarum fimbriatum" in by_name, True)
check("  …with an empty common list",
      by_name["Agarum fimbriatum"]["common"], [])
check("apostrophe does not confuse the split",
      by_name["Codium fragile"]["common"], ["dead man's fingers"])
check("bare binomial kept", "Chromodoris annae" in by_name, True)
check("single-common-name leaf", by_name["Turbinaria turbinata"]["common"],
      ["blistered saucer leaf algae"])

# Path is the ancestor chain, and must not include the leaf itself.
path = by_name["Fucus vesiculosus"]["path"]
check("path starts at the root", path[0], "Marine Life")
check("path keeps the nesting", path[-1], "Brown Algae: Brown Seaweeds: Phaeophyta")
check("path excludes the species leaf", "Fucus vesiculosus" in path, False)

# Same species listed twice collapses to one record rather than producing a duplicate.
check("duplicate species collapsing", len([r for r in recs
      if r["scientific"] == "Chromodoris annae"]), 1)

print(f"\nPIPELINE PARSE: {checks - failed}/{checks} checks pass")
if failed:
    print(f"PIPELINE PARSE: {failed} FAILED")
    sys.exit(1)
print("PIPELINE PARSE: ALL PASS")
